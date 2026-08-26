#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def quat_to_6d(quat_xyzw):
    """Convert a quaternion [x, y, z, w] to a 6D rotation representation."""
    quat = np.asarray(quat_xyzw, dtype=np.float32)
    if quat.size != 4:
        return np.zeros(6, dtype=np.float32)

    x, y, z, w = quat / (np.linalg.norm(quat) + 1e-12)
    rot = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return np.concatenate([rot[:, 0], rot[:, 1]], axis=0).astype(np.float32)


def extract_state(low_dim):
    """Build a compact robot state from the raw recorded low_dim.npy."""
    if isinstance(low_dim, np.ndarray) and low_dim.shape == ():
        low_dim = low_dim.item()

    joint_pos = np.asarray(low_dim.get("joints", {}).get("position", []), dtype=np.float32).reshape(-1)
    ee_pos = np.asarray(low_dim.get("ee_position", []), dtype=np.float32).reshape(-1)
    ee_ori = np.asarray(low_dim.get("ee_orientation", []), dtype=np.float32).reshape(-1)

    joint_state = joint_pos[:7] if joint_pos.size >= 7 else np.zeros(7, dtype=np.float32)
    pos_state = ee_pos[:3] if ee_pos.size >= 3 else np.zeros(3, dtype=np.float32)

    if ee_ori.size == 4:
        ori_state = quat_to_6d(ee_ori)
    elif ee_ori.size == 6:
        ori_state = ee_ori[:6]
    elif ee_ori.size == 3:
        ori_state = np.zeros(6, dtype=np.float32)
        ori_state[:3] = ee_ori[:3]
    else:
        ori_state = np.zeros(6, dtype=np.float32)

    state = np.concatenate([joint_state, pos_state, ori_state[:6]], axis=0).astype(np.float32)
    return state


def load_raw_episode(episode_dir: Path):
    frame_dirs = sorted(episode_dir.iterdir(), key=lambda p: p.name.isdigit())
    frame_dirs = [p for p in frame_dirs if p.is_dir() and p.name.isdigit()]
    frame_dirs = sorted(frame_dirs, key=lambda p: int(p.name))

    if not frame_dirs:
        return None

    images = []
    states = []
    actions = []

    for i, frame_dir in enumerate(frame_dirs):
        img_path = frame_dir / "frame.png"
        low_path = frame_dir / "low_dim.npy"

        if not img_path.exists() or not low_path.exists():
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        low_dim = np.load(low_path, allow_pickle=True)
        if isinstance(low_dim, np.ndarray) and low_dim.shape == ():
            low_dim = low_dim.item()

        state = extract_state(low_dim)
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        states.append(state)

        if i + 1 < len(frame_dirs):
            next_low = np.load(frame_dirs[i + 1] / "low_dim.npy", allow_pickle=True)
            if isinstance(next_low, np.ndarray) and next_low.shape == ():
                next_low = next_low.item()
            next_state = extract_state(next_low)
            actions.append(next_state - state)
        else:
            actions.append(np.zeros_like(state))

    if len(images) == 0:
        return None

    images_arr = np.stack(images, axis=0).astype(np.uint8)
    states_arr = np.stack(states, axis=0).astype(np.float32)
    actions_arr = np.stack(actions, axis=0).astype(np.float32)
    return {"images": images_arr, "states": states_arr, "actions": actions_arr}


def convert_dataset(dataset_dir: Path, output_dir: Path):
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    episodes = sorted([p for p in dataset_dir.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))
    if not episodes:
        raise ValueError(f"No episode folders found under {dataset_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset_name": dataset_dir.name,
        "num_episodes": 0,
        "episodes": [],
        "state_dim": 0,
        "action_dim": 0,
        "image_shape": [0, 0, 0, 3],
    }

    for ep_idx, episode_dir in enumerate(episodes):
        episode_data = load_raw_episode(episode_dir)
        if episode_data is None:
            continue

        episode_file = output_dir / f"episode_{ep_idx:04d}.npz"
        np.savez_compressed(
            episode_file,
            images=episode_data["images"],
            states=episode_data["states"],
            actions=episode_data["actions"],
        )

        manifest["episodes"].append({
            "episode_index": ep_idx,
            "file": str(episode_file.name),
            "num_steps": int(episode_data["images"].shape[0]),
        })

    if not manifest["episodes"]:
        raise ValueError(f"No valid episodes were converted from {dataset_dir}")

    sample = np.load(output_dir / manifest["episodes"][0]["file"], allow_pickle=False)
    manifest["state_dim"] = int(sample["states"].shape[1])
    manifest["action_dim"] = int(sample["actions"].shape[1])
    manifest["image_shape"] = list(sample["images"].shape)
    manifest["num_episodes"] = len(manifest["episodes"])

    with open(output_dir / "pi0_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Converted {manifest['num_episodes']} episodes to {output_dir}")
    print(f"State dim: {manifest['state_dim']}, Action dim: {manifest['action_dim']}")
    print(f"Image shape: {manifest['image_shape']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a unified robot dataset into a Pi0-style VLA dataset.")
    parser.add_argument("dataset_name", help="Dataset name under kinova-diffusion/data, e.g. unified")
    parser.add_argument("--dataset-root", default=None, help="Optional dataset root. Defaults to kinova-diffusion/data")
    parser.add_argument("--output-dir", default=None, help="Optional output directory. Defaults to pi0_datasets/<dataset_name>")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    dataset_root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else repo_root / "kinova-diffusion" / "data"
    dataset_dir = dataset_root / args.dataset_name

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else repo_root / "pi0_datasets" / args.dataset_name
    convert_dataset(dataset_dir, output_dir)
