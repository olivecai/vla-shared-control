#!/usr/bin/env python3
"""
LoRA fine-tune OpenVLA (https://github.com/openvla/openvla) on episodes recorded by record.py.

This is a minimal, from-scratch training loop -- it does NOT use OpenVLA's official RLDS /
TensorFlow-Datasets pipeline. It reads record.py's plain per-episode folders directly with
PyTorch instead, which keeps the whole pipeline easy to read and debug at the cost of matching
the original training setup exactly (single GPU, no dataset mixing, no data augmentation).

Action space: each training example predicts the DELTA in joint angles + gripper position between
one recorded step and the next (7 joints + 1 gripper = 8 values), normalized to [-1, 1] using the
1st/99th percentile of the recorded deltas -- the same normalization convention OpenVLA uses.

Requires a CUDA GPU. bf16 LoRA fine-tuning of the 7B model fits on a single ~24GB GPU.
See requirements.txt for the extra (non-ROS) Python packages this needs.

Usage:
    python3 vla/training_scripts/train_openvla.py --data-dir vla/data --output-dir vla/checkpoints/kinova-lora
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model

MODEL_ID = "openvla/openvla-7b"
ACTION_DIM = 8   # 7 Kinova Gen3 joints + 1 gripper position
N_BINS = 256     # number of discrete action bins -- matches OpenVLA's pretraining setup

# LoRA targets the attention + MLP projections of OpenVLA's Llama-2 language backbone,
# matching OpenVLA's own official fine-tuning recipe.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def discretize(action, bins):
    """Map a continuous action in [-1, 1] to an integer bin in [0, N_BINS - 1]."""
    action = np.clip(action, -1.0, 1.0)
    return np.digitize(action, bins) - 1


class KinovaEpisodeDataset(Dataset):
    """
    Every (episode, timestep t) becomes one training example:
        input  = (image_t, language instruction)
        target = normalized delta (joint angles + gripper) executed between step t and t+1
    """

    def __init__(self, data_dir, cam_id=0):
        self.samples = []    # list of (episode_dir, t)
        self.episodes = {}   # episode_dir -> {"positions": np.ndarray [T, ACTION_DIM], "instruction": str}
        self.cam_id = cam_id

        for ep_dir in sorted(glob.glob(os.path.join(data_dir, "episode_*"))):
            joints = np.load(os.path.join(ep_dir, "joint_positions.npy"))          # (T, 7)
            gripper = np.load(os.path.join(ep_dir, "gripper_positions.npy"))       # (T,)
            positions = np.concatenate([joints, gripper[:, None]], axis=1)         # (T, 8)
            with open(os.path.join(ep_dir, "instruction.txt")) as f:
                instruction = f.read().strip()
            self.episodes[ep_dir] = {"positions": positions, "instruction": instruction}
            for t in range(len(positions) - 1):  # need a "next" state to form an action
                self.samples.append((ep_dir, t))

        if not self.samples:
            raise RuntimeError(f"No episodes with >=2 steps found in {data_dir}. Record some data first.")

        deltas = np.concatenate(
            [ep["positions"][1:] - ep["positions"][:-1] for ep in self.episodes.values()], axis=0
        )
        self.q01 = np.percentile(deltas, 1, axis=0)
        self.q99 = np.percentile(deltas, 99, axis=0)

    def __len__(self):
        return len(self.samples)

    def normalize(self, delta):
        return np.clip(2 * (delta - self.q01) / (self.q99 - self.q01 + 1e-8) - 1, -1.0, 1.0)

    def __getitem__(self, idx):
        ep_dir, t = self.samples[idx]
        ep = self.episodes[ep_dir]
        image_path = os.path.join(ep_dir, f"cam{self.cam_id}", f"{t:06d}.jpg")
        image = Image.open(image_path).convert("RGB")
        delta = ep["positions"][t + 1] - ep["positions"][t]
        return image, ep["instruction"], self.normalize(delta)

    def save_statistics(self, path, unnorm_key):
        """
        Save normalization stats in the format OpenVLA's predict_action() expects at inference
        time (it looks for a dataset_statistics.json next to the model checkpoint).
        """
        stats = {unnorm_key: {"action": {
            "q01": self.q01.tolist(),
            "q99": self.q99.tolist(),
            "mask": [True] * ACTION_DIM,
        }}}
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)


def build_batch(examples, processor, tokenizer, bins):
    """
    Turn a list of (image, instruction, action) examples into a padded training batch.

    Each example's target text is the prompt below followed by ACTION_DIM tokens, one per
    action dimension, taken from the END of the vocabulary (mirrors OpenVLA's own
    ActionTokenizer: token_id = vocab_size - 1 - bin_index).
    """
    vocab_size = tokenizer.vocab_size
    input_ids_list, labels_list, pixel_values_list = [], [], []

    for image, instruction, action in examples:
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = processor(prompt, image, return_tensors="pt")
        prompt_ids = inputs["input_ids"][0]
        pixel_values_list.append(inputs["pixel_values"][0])

        action_bins = discretize(action, bins)
        action_token_ids = torch.tensor(vocab_size - 1 - action_bins, dtype=torch.long)
        eos_id = torch.tensor([tokenizer.eos_token_id], dtype=torch.long)

        input_ids_list.append(torch.cat([prompt_ids, action_token_ids, eos_id]))
        # Only the action tokens (+EOS) contribute to the loss -- mask out the prompt.
        labels_list.append(torch.cat([torch.full_like(prompt_ids, -100), action_token_ids, eos_id]))

    max_len = max(len(ids) for ids in input_ids_list)
    pad_id = tokenizer.pad_token_id or 0
    input_ids = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
    for i, (ids, lab) in enumerate(zip(input_ids_list, labels_list)):
        input_ids[i, :len(ids)] = ids
        labels[i, :len(lab)] = lab
        attention_mask[i, :len(ids)] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": torch.stack(pixel_values_list),
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="vla/data")
    parser.add_argument("--output-dir", default="vla/checkpoints/kinova-lora")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--unnorm-key", default="kinova")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    dataset = KinovaEpisodeDataset(args.data_dir, cam_id=args.cam_id)
    print(f"Loaded {len(dataset)} training steps from {len(dataset.episodes)} episodes.")

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer = processor.tokenizer
    bins = np.linspace(-1.0, 1.0, N_BINS)

    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to("cuda")

    lora_cfg = LoraConfig(
        r=32, lora_alpha=16, lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES, init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_cfg)
    vla.print_trainable_parameters()

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda batch: build_batch(batch, processor, tokenizer, bins),
    )
    optimizer = torch.optim.AdamW(vla.parameters(), lr=args.lr)

    vla.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)

            out = vla(**batch)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                print(f"epoch {epoch} step {step}: loss {out.loss.item():.4f}")

    print(f"Merging LoRA weights and saving to {args.output_dir} ...")
    os.makedirs(args.output_dir, exist_ok=True)
    merged = vla.merge_and_unload()  # bake LoRA weights in so predict_action() works directly at inference
    merged.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    dataset.save_statistics(os.path.join(args.output_dir, "dataset_statistics.json"), args.unnorm_key)
    print("Done.")


if __name__ == "__main__":
    main()
