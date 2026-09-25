#!/usr/bin/env python3
"""
Convert record.py's per-episode dataset into a LeRobot v2.1 dataset compatible with
NVIDIA Isaac GR00T fine-tuning (https://github.com/NVIDIA/Isaac-GR00T).

Input layout (record.py's output):
    <data_dir>/episode_<timestamp>/
        instruction.txt
        joint_positions.npy    # (T, 7) float32
        gripper_positions.npy  # (T,) float32
        timestamps.npy         # (T,) float64, wall-clock seconds
        cam{id}/000000.jpg ...

Output layout (LeRobot v2.1 + GR00T's extra modality.json), verified against NVIDIA's own
demo_data/cube_to_bowl_5 example dataset:
    <output_dir>/
        meta/{info.json, modality.json, tasks.jsonl, episodes.jsonl, stats.json}
        data/chunk-000/episode_XXXXXX.parquet
        videos/chunk-000/observation.images.cam{id}/episode_XXXXXX.mp4

ACTION DEFINITION -- read before training on this:
    record.py only logs *observed* robot state (joint angles + gripper), not a distinct
    commanded action. Following LeRobot/GR00T convention for observation-only data, this
    script defines action[t] = observation.state[t+1] (the NEXT observed state), i.e. the
    model is trained to predict where the arm goes next. This is an ABSOLUTE target, not
    the normalized DELTA that vla/train.py's from-scratch OpenVLA loop uses -- GR00T does
    its own per-field normalization from meta/stats.json, so don't pre-normalize here.

FPS: record.py's timestamps.npy has real per-frame wall-clock times (used for the
    per-frame `timestamp` column below), but the *nominal* fps burned into info.json and
    each video (LeRobot assumes constant-rate video) is just whatever --fps you pass here
    -- default matches record.py's own --hz default (5).

Requires: pandas, pyarrow, Pillow, and the `ffmpeg` binary on PATH.
    pip install pandas pyarrow pillow

Usage:
    python3 vla/offline_scripts/convert_to_lerobot_groot.py \
        --data-dir vla/data --output-dir vla/lerobot_data


        for example:
        python3 vla/offline_scripts/convert_to_lerobot_groot.py  --data-dir vla/data_sep20 --output-dir vla/lerobo
t_data
Converted vla/data_sep20/episode_1790037263 -> episode_000000 (89 frames, cams [0])
Converted vla/data_sep20/episode_1790037305 -> episode_000001 (84 frames, cams [0])
Converted vla/data_sep20/episode_1790037341 -> episode_000002 (89 frames, cams [0])
"""
import argparse
import glob
import json
import os
import subprocess

import numpy as np

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit("Missing dependency. Install with: pip install pandas pyarrow pillow") from exc

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("Missing dependency. Install with: pip install pandas pyarrow pillow") from exc

ACTION_NAMES = [f"joint_{i}" for i in range(7)] + ["gripper"]
CHUNKS_SIZE = 1000


def find_cam_ids(ep_dir):
    cam_ids = []
    for name in sorted(os.listdir(ep_dir)):
        if name.startswith("cam") and os.path.isdir(os.path.join(ep_dir, name)):
            cam_ids.append(int(name[len("cam"):]))
    return sorted(cam_ids)


def image_size(cam_dir):
    first = sorted(glob.glob(os.path.join(cam_dir, "*.jpg")))[0]
    with Image.open(first) as im:
        return im.height, im.width


def encode_video(cam_dir, out_path, fps):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_frames = len(glob.glob(os.path.join(cam_dir, "*.jpg")))
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps), "-i", os.path.join(cam_dir, "%06d.jpg"),
        "-frames:v", str(n_frames),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        out_path,
    ]
    subprocess.run(cmd, check=True)


def chunk_dir(base, chunk_idx):
    return os.path.join(base, f"chunk-{chunk_idx:03d}")


def stats_block(arr):
    return {
        "mean": np.mean(arr, axis=0).tolist(),
        "std": np.std(arr, axis=0).tolist(),
        "min": np.min(arr, axis=0).tolist(),
        "max": np.max(arr, axis=0).tolist(),
        "q01": np.percentile(arr, 1, axis=0).tolist(),
        "q99": np.percentile(arr, 99, axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="vla/data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, default=5.0, help="Nominal fps, matches record.py's --hz default")
    parser.add_argument("--robot-type", default="kinova_gen3")
    args = parser.parse_args()

    ep_dirs = sorted(glob.glob(os.path.join(args.data_dir, "episode_*")))
    if not ep_dirs:
        raise SystemExit(f"No episode_* folders found in {args.data_dir}")

    ref_cam_ids = find_cam_ids(ep_dirs[0])
    if not ref_cam_ids:
        raise SystemExit(f"{ep_dirs[0]} has no cam* subfolders")

    os.makedirs(os.path.join(args.output_dir, "meta"), exist_ok=True)

    task_index_by_instruction = {}
    episodes_meta = []
    all_states, all_actions, all_timestamps = [], [], []
    cam_video_info = {}  # cam_id -> (height, width), filled in from the first episode that has it
    out_ep_idx = 0
    total_frames = 0

    for ep_dir in ep_dirs:
        joints = np.load(os.path.join(ep_dir, "joint_positions.npy")).astype(np.float32)     # (T, 7)
        gripper = np.load(os.path.join(ep_dir, "gripper_positions.npy")).astype(np.float32)   # (T,)
        timestamps = np.load(os.path.join(ep_dir, "timestamps.npy")).astype(np.float64)       # (T,)
        with open(os.path.join(ep_dir, "instruction.txt")) as f:
            instruction = f.read().strip()

        T = len(joints)
        if T < 2:
            print(f"Skipping {ep_dir}: only {T} step(s), need >=2 to form an action")
            continue

        cam_ids = find_cam_ids(ep_dir)
        if cam_ids != ref_cam_ids:
            raise SystemExit(
                f"{ep_dir} has cams {cam_ids}, but {ep_dirs[0]} has {ref_cam_ids}. "
                "All episodes must record the same set of cameras to convert together."
            )

        state = np.concatenate([joints, gripper[:, None]], axis=1)  # (T, 8)
        # action[t] = next observed state -- see ACTION DEFINITION in the module docstring.
        obs_state = state[:-1]
        action = state[1:]
        n_out = T - 1

        if instruction not in task_index_by_instruction:
            task_index_by_instruction[instruction] = len(task_index_by_instruction)
        task_index = task_index_by_instruction[instruction]

        chunk_idx = out_ep_idx // CHUNKS_SIZE
        ep_name = f"episode_{out_ep_idx:06d}"

        for cam_id in cam_ids:
            cam_dir = os.path.join(ep_dir, f"cam{cam_id}")
            h, w = image_size(cam_dir)
            cam_video_info.setdefault(cam_id, (h, w))
            video_dir = os.path.join(
                chunk_dir(os.path.join(args.output_dir, "videos"), chunk_idx),
                f"observation.images.cam{cam_id}",
            )
            encode_video(cam_dir, os.path.join(video_dir, f"{ep_name}.mp4"), args.fps)

        df = pd.DataFrame({
            "observation.state": list(obs_state),
            "action": list(action),
            "timestamp": (timestamps[:-1] - timestamps[0]).astype(np.float32),
            "frame_index": np.arange(n_out, dtype=np.int64),
            "episode_index": np.full(n_out, out_ep_idx, dtype=np.int64),
            "index": np.arange(total_frames, total_frames + n_out, dtype=np.int64),
            "task_index": np.full(n_out, task_index, dtype=np.int64),
        })
        data_dir = chunk_dir(os.path.join(args.output_dir, "data"), chunk_idx)
        os.makedirs(data_dir, exist_ok=True)
        df.to_parquet(os.path.join(data_dir, f"{ep_name}.parquet"), index=False)

        episodes_meta.append({"episode_index": out_ep_idx, "tasks": [instruction], "length": n_out})
        all_states.append(obs_state)
        all_actions.append(action)
        all_timestamps.append(df["timestamp"].to_numpy()[:, None])
        total_frames += n_out
        out_ep_idx += 1
        print(f"Converted {ep_dir} -> {ep_name} ({n_out} frames, cams {cam_ids})")

    if out_ep_idx == 0:
        raise SystemExit("No episodes had >=2 steps -- nothing to convert.")

    with open(os.path.join(args.output_dir, "meta", "tasks.jsonl"), "w") as f:
        for instruction, idx in sorted(task_index_by_instruction.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": idx, "task": instruction}) + "\n")

    with open(os.path.join(args.output_dir, "meta", "episodes.jsonl"), "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    stats = {
        "action": stats_block(np.concatenate(all_actions, axis=0)),
        "observation.state": stats_block(np.concatenate(all_states, axis=0)),
        "timestamp": stats_block(np.concatenate(all_timestamps, axis=0)),
    }
    with open(os.path.join(args.output_dir, "meta", "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    features = {
        "action": {"dtype": "float32", "shape": [8], "names": ACTION_NAMES},
        "observation.state": {"dtype": "float32", "shape": [8], "names": ACTION_NAMES},
    }
    for cam_id in ref_cam_ids:
        h, w = cam_video_info[cam_id]
        features[f"observation.images.cam{cam_id}"] = {
            "dtype": "video",
            "shape": [h, w, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": h, "video.width": w,
                "video.codec": "h264", "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False, "video.fps": args.fps,
                "video.channels": 3, "has_audio": False,
            },
        }
    for key, dtype in [("timestamp", "float32"), ("frame_index", "int64"),
                        ("episode_index", "int64"), ("index", "int64"), ("task_index", "int64")]:
        features[key] = {"dtype": dtype, "shape": [1], "names": None}

    info = {
        "codebase_version": "v2.1",
        "robot_type": args.robot_type,
        "total_episodes": out_ep_idx,
        "total_frames": total_frames,
        "total_tasks": len(task_index_by_instruction),
        "chunks_size": CHUNKS_SIZE,
        "fps": args.fps,
        "splits": {"train": f"0:{out_ep_idx}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "total_chunks": (out_ep_idx - 1) // CHUNKS_SIZE,
        "total_videos": out_ep_idx * len(ref_cam_ids),
    }
    with open(os.path.join(args.output_dir, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    modality = {
        "state": {"arm": {"start": 0, "end": 7}, "gripper": {"start": 7, "end": 8}},
        "action": {"arm": {"start": 0, "end": 7}, "gripper": {"start": 7, "end": 8}},
        "video": {f"cam{cam_id}": {"original_key": f"observation.images.cam{cam_id}"} for cam_id in ref_cam_ids},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    with open(os.path.join(args.output_dir, "meta", "modality.json"), "w") as f:
        json.dump(modality, f, indent=4)

    print(f"\nDone: {out_ep_idx} episodes, {total_frames} frames -> {args.output_dir}")


if __name__ == "__main__":
    main()
