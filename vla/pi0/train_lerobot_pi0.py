#!/usr/bin/env python3
"""

INSPECT local lerobot v2 dataset
(use convert_to_lerobot_v2.py to produce dataset)


Usage:
    # See the exact command that would run, without running it:
    python3 train_lerobot_pi0.py /path/to/lerobot_dataset --dry-run

    # Actually launch training:
    python3 train_lerobot_pi0.py /path/to/lerobot_dataset \
        --repo-id my_kinova_dataset \
        --exp-name my_run \
        --steps 30000 \
        --batch-size 8

Requires: pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git"
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import cv2
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependencies. Install with: pip install pandas pyarrow opencv-python-headless"
    ) from exc


# pi0/pi0.5 base checkpoints were pretrained with 3 cameras + 32-dim state/action.
PRETRAINED_CAMERA_SLOTS = 3
PRETRAINED_IMAGE_SIZE = 224


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_first_parquet(dataset_root: Path) -> Path:
    candidates = sorted((dataset_root / "data").rglob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")
    return candidates[0]


def find_first_video(dataset_root: Path) -> Optional[Path]:
    videos_dir = dataset_root / "videos"
    if not videos_dir.exists():
        return None
    candidates = sorted(videos_dir.rglob("*.mp4"))
    return candidates[0] if candidates else None


def vector_len(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset parquet (columns: {list(df.columns)})")
    first_non_null = df[column].dropna()
    if first_non_null.empty:
        raise ValueError(f"Column '{column}' is present but empty in the sampled parquet file")
    value = first_non_null.iloc[0]
    return len(value)


def image_dims(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video to inspect resolution: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Got invalid resolution ({width}x{height}) from {video_path}")
    return width, height


def image_feature_names(dataset_root: Path) -> list[str]:
    videos_dir = dataset_root / "videos"
    if not videos_dir.exists():
        return []
    # e.g. videos/chunk-000/observation.images.cam_high/episode_000000.mp4
    names = set()
    for chunk_dir in videos_dir.iterdir():
        if not chunk_dir.is_dir():
            continue
        for cam_dir in chunk_dir.iterdir():
            if cam_dir.is_dir():
                names.add(cam_dir.name)
    return sorted(names)


def check_lerobot_installed() -> None:
    if shutil.which("lerobot-train") is None:
        raise SystemExit(
            "lerobot-train not found on PATH. Install with:\n"
            '  pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git"'
        )


def build_input_features(
    image_names: list[str],
    img_width: int,
    img_height: int,
    state_dim: int,
    empty_cameras: int,
) -> str:
    features: Dict[str, Any] = {}
    for name in image_names:
        features[name] = {"shape": [3, img_height, img_width], "type": "VISUAL"}
    features["observation.state"] = {"shape": [state_dim], "type": "STATE"}
    return json.dumps(features)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch lerobot-train (pi0/pi0.5) against a local LeRobot v2 dataset.")
    parser.add_argument("dataset_root", type=Path, help="Path to the LeRobot v2 dataset (contains meta/, data/, videos/)")
    parser.add_argument("--repo-id", default=None, help="Dataset repo id to pass to --dataset.repo_id (defaults to the dataset folder name)")
    parser.add_argument("--policy-type", choices=["pi0", "pi05"], default="pi0")
    parser.add_argument("--pretrained-path", default=None, help="Defaults to lerobot/pi0_base or lerobot/pi05_base depending on --policy-type")
    parser.add_argument("--exp-name", default="lerobot_finetune", help="Experiment name / output subdirectory")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--train-expert-only", action="store_true", help="Freeze the VLM, train only the action expert (lower memory)")
    parser.add_argument("--empty-cameras", type=int, default=None, help="Override number of padding camera slots; default is auto-computed against a 3-camera pretrained checkpoint")
    parser.add_argument("--extra-arg", action="append", default=[], help="Additional raw args to pass through to lerobot-train, repeatable, e.g. --extra-arg=--save_freq=5000")
    parser.add_argument("--dry-run", action="store_true", help="Print the command instead of running it")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")

    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise SystemExit(
            f"No meta/info.json found under {dataset_root}. This doesn't look like a "
            f"LeRobot v2 dataset -- did you run the conversion script first?"
        )
    info = load_json(info_path)

    parquet_path = find_first_parquet(dataset_root)
    df = pd.read_parquet(parquet_path)

    state_dim = vector_len(df, "observation.state")
    action_dim = vector_len(df, "action")

    image_names = image_feature_names(dataset_root)
    if not image_names:
        raise SystemExit(
            f"No video feature directories found under {dataset_root / 'videos'}. "
            f"pi0/pi0.5 require at least one camera; this dataset appears to have none."
        )

    video_path = find_first_video(dataset_root)
    img_width, img_height = image_dims(video_path)

    empty_cameras = args.empty_cameras
    if empty_cameras is None:
        empty_cameras = max(0, PRETRAINED_CAMERA_SLOTS - len(image_names))

    pretrained_path = args.pretrained_path
    if pretrained_path is None:
        pretrained_path = f"lerobot/{'pi05_base' if args.policy_type == 'pi05' else 'pi0_base'}"

    repo_id = args.repo_id or dataset_root.name

    print("--- Detected dataset properties ---")
    print(f"  parquet sampled:       {parquet_path.relative_to(dataset_root)}")
    print(f"  observation.state dim: {state_dim}")
    print(f"  action dim:            {action_dim}")
    print(f"  image feature(s):      {image_names}")
    print(f"  image resolution:      {img_width}x{img_height}")
    if img_width != PRETRAINED_IMAGE_SIZE or img_height != PRETRAINED_IMAGE_SIZE:
        print(
            f"  NOTE: pretrained checkpoint expects {PRETRAINED_IMAGE_SIZE}x{PRETRAINED_IMAGE_SIZE} "
            f"images. lerobot-train resizes internally, but confirm your camera's aspect ratio "
            f"isn't badly distorted by that resize before trusting results."
        )
    if action_dim != state_dim:
        print(
            f"  NOTE: action dim ({action_dim}) != state dim ({state_dim}). This is expected "
            f"here since action = next EE pose only, while state also includes joint angles "
            f"and (if present) segmentation centroids -- not a bug, just worth knowing."
        )
    print(f"  empty_cameras to pad:  {empty_cameras} (pretrained expects {PRETRAINED_CAMERA_SLOTS} total)")
    print()

    input_features = build_input_features(image_names, img_width, img_height, state_dim, empty_cameras)

    cmd = [
        "lerobot-train",
        f"--policy.type={args.policy_type}",
        f"--policy.pretrained_path={pretrained_path}",
        f"--policy.device={args.device}",
        f"--policy.dtype={args.dtype}",
        f"--policy.empty_cameras={empty_cameras}",
        f"--policy.input_features={input_features}",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_root}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--job_name={args.exp_name}",
    ]
    if args.train_expert_only:
        cmd.append("--policy.train_expert_only=true")
    cmd.extend(args.extra_arg)

    print("--- Command ---")
    print(" \\\n    ".join(cmd))
    print()

    if args.dry_run:
        print("(dry run -- not executing)")
        return

    check_lerobot_installed()
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()