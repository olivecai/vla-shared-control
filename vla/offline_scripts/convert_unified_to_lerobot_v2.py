#!/usr/bin/env python3
"""Convert a unified dataset into a LeRobot v2 dataset layout for pi0-style VLA training.

Expected raw input layout (output of record.py in "unified" or "vla" mode):
    <dataset>/
    └── <episode_index>/            # e.g. 0, 1, 2, ...
        └── <frame_index>/          # e.g. 0, 1, 2, ...
            ├── low_dim.npy
            ├── frame.png           # RGB, present for vla/unified models
            ├── seg_pc.npy          # optional, present for hitl_hgd/unified
            └── depth.npy

Also accepts an already-LeRobot-formatted dataset (has a meta/ directory) and
re-normalizes it into the same output layout.

Output mirrors the LeRobot v2 dataset layout used by pi0 and other VLA pipelines:
    <out_dir>/
    ├── meta/
    │   ├── info.json
    │   ├── episodes.jsonl
    │   ├── tasks.jsonl
    │   └── stats.safetensors
    ├── data/
    │   └── chunk-000/
    │       ├── episode_000000.parquet
    │       └── episode_000001.parquet
    └── videos/
        └── chunk-000/
            └── observation.images.cam_high/
                └── episode_000000.mp4

IMPORTANT — read before running on a real dataset:

1. ACTION DEFINITION. record.py's low_dim.npy only stores *observed* robot
   state (joint state, EE pose) at each timestep — it does not store the
   commanded control signal that was actually sent to the arm. In the
   absence of a true commanded-action field, this script defines
   "action" at frame i as the *next* observed EE pose (frame i+1), so the
   model is trained to predict where the arm goes next, not to reproduce
   its own current state. This is a reasonable proxy for behavior cloning
   from observed trajectories, but it is not the same as recording true
   commanded actions. If record.py can be updated to log the actual
   commanded target (whatever is sent to png_control/cartesian_control/
   joint_control) under a distinct key, prefer that — see
   --action-source below.

2. LANGUAGE TASK. Raw low_dim.npy has no language instruction. This script
   will NOT silently emit "task": "" for every episode — pass --task or
   --tasks-file, or explicitly opt in to blank tasks with
   --allow-empty-task if you really intend to produce a non-language-
   conditioned dataset.

3. FPS. record.py does not currently timestamp low_dim.npy entries, so
   there is no ground-truth capture rate to derive from the data. --fps
   is a nominal value (default 30) burned into meta/info.json; it is not
   measured. If you need accurate per-frame timing, add a timestamp field
   to low_dim.npy in record.py and update this script to consume it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover - exercised only at runtime
    raise SystemExit(
        "Missing Python dependencies for this script. Install them with: "
        "pip install numpy pandas pyarrow opencv-python safetensors"
    ) from exc

try:
    from safetensors.numpy import save_file as save_safetensors_file
except ImportError:  # pragma: no cover - optional dependency for stats export
    save_safetensors_file = None


DEFAULT_TASK_KEYS = ["task", "instruction", "language_instruction", "prompt", "text"]
MAX_FRAMES_PER_EPISODE = 512  # matches record.py's vla/unified cap


# --------------------------------------------------------------------------
# generic helpers
# --------------------------------------------------------------------------

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (list, tuple)):
        return [to_python_scalar(v) for v in value]
    return value


def find_task_value(record: Dict[str, Any]) -> str:
    for key in DEFAULT_TASK_KEYS:
        if key in record and record[key] not in (None, ""):
            return str(record[key])
    return ""


def write_stats_safetensors(path: Path, stats_by_key: Dict[str, np.ndarray]) -> None:
    if save_safetensors_file is None:
        raise RuntimeError("safetensors is not installed. Install it with: pip install safetensors")
    tensor_dict = {key: np.asarray(value, dtype=np.float32) for key, value in stats_by_key.items()}
    save_safetensors_file(tensor_dict, path)


def coerce_vector_column(series: pd.Series) -> np.ndarray:
    vecs = []
    for value in series.tolist():
        if value is None:
            vecs.append(np.zeros(0, dtype=np.float32))
            continue
        arr = np.asarray(value, dtype=np.float32)
        vecs.append(arr.reshape(-1) if arr.ndim else np.asarray([float(arr)], dtype=np.float32))
    return np.asarray(vecs, dtype=np.float32)


def collect_stats_from_episode(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    stats: Dict[str, np.ndarray] = {}
    for column in ["observation.state", "action"]:
        if column in df.columns:
            arr = coerce_vector_column(df[column]).astype(np.float32)
            if arr.size > 0:
                stats[f"{column}_mean"] = arr.mean(axis=0)
                std = arr.std(axis=0)
                stats[f"{column}_std"] = np.where(std < 1e-8, 1.0, std)
    return stats


def write_video_from_frames(frame_paths: List[Path], output_mp4: Path, fps: float) -> int:
    """Writes frames in the given order. Returns the number of frames actually written.

    Caller is responsible for ensuring frame_paths is already aligned 1:1 with the
    parquet rows for this episode -- this function does not skip/reorder anything.
    """
    if not frame_paths:
        return 0
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Could not read first frame: {frame_paths[0]}")
    height, width = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    ensure_dir(output_mp4.parent)
    writer = cv2.VideoWriter(str(output_mp4), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_mp4}")
    written = 0
    try:
        for path in frame_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(
                    f"Frame unreadable mid-episode: {path}. Refusing to write a video "
                    f"with a silently dropped/duplicated frame."
                )
            writer.write(frame)
            written += 1
    finally:
        writer.release()
    return written


def copy_video_tree(src_root: Path, dst_root: Path) -> None:
    src_videos = src_root / "videos"
    if not src_videos.exists():
        return
    dst_videos = dst_root / "videos"
    for src in sorted(src_videos.rglob("*")):
        if src.is_dir():
            continue
        rel = src.relative_to(src_videos)
        dst = dst_videos / rel
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


# --------------------------------------------------------------------------
# raw unified dataset -> lerobot v2
# --------------------------------------------------------------------------

def extract_ee_pose(state_info: Dict[str, Any]) -> np.ndarray:
    """Position + orientation only -- this is the quantity we use as the action target."""
    values: List[float] = []
    ee_pos = state_info.get("ee_position", [])
    if isinstance(ee_pos, (list, tuple, np.ndarray)):
        values.extend(np.asarray(ee_pos, dtype=np.float32).reshape(-1).tolist())
    ee_ori = state_info.get("ee_orientation", [])
    if isinstance(ee_ori, (list, tuple, np.ndarray)):
        values.extend(np.asarray(ee_ori, dtype=np.float32).reshape(-1).tolist())
    if not values:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def flatten_robot_state(state_info: Dict[str, Any]) -> np.ndarray:
    """Full observation vector: joints + EE pose + (optional) segmentation centroids."""
    values: List[float] = []
    joints = state_info.get("joints", {})
    pos = joints.get("position", [])
    if isinstance(pos, (list, tuple, np.ndarray)):
        values.extend(np.asarray(pos, dtype=np.float32).reshape(-1).tolist())
    values.extend(extract_ee_pose(state_info).tolist())
    centroids = state_info.get("centroids", [])
    if isinstance(centroids, (list, tuple, np.ndarray)):
        values.extend(np.asarray(centroids, dtype=np.float32).reshape(-1).tolist())
    if not values:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def detect_raw_unified_dataset(input_dir: Path) -> bool:
    if not input_dir.exists() or (input_dir / "meta").exists():
        return False
    return any(child.is_dir() and child.name.isdigit() for child in input_dir.iterdir())


def dataset_has_any_rgb(input_dir: Path, episode_dirs: List[Path]) -> bool:
    """Peek across episodes (not just the first) for at least one frame.png.

    A single empty/failed episode-0 shouldn't cause a false negative, and
    conversely we don't want to claim a video feature exists for a dataset
    that never actually recorded RGB (e.g. hitl_d / hitl_hgd captures,
    which also produce numeric episode folders).
    """
    for episode_dir in episode_dirs:
        for timestep_dir in episode_dir.iterdir():
            if timestep_dir.is_dir() and (timestep_dir / "frame.png").exists():
                return True
    return False


def resolve_task_for_episode(
    episode_dir_name: str,
    episode_index: int,
    tasks_map: Dict[str, str],
    global_task: Optional[str],
) -> str:
    if episode_dir_name in tasks_map:
        return tasks_map[episode_dir_name]
    if str(episode_index) in tasks_map:
        return tasks_map[str(episode_index)]
    if global_task is not None:
        return global_task
    return ""


def convert_raw_unified_dataset(
    input_dir: Path,
    output_dir: Path,
    repo_id: Optional[str],
    fps: float,
    global_task: Optional[str],
    tasks_map: Dict[str, str],
    allow_empty_task: bool,
    last_frame_action: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_meta_dir = output_dir / "meta"
    ensure_dir(output_meta_dir)
    ensure_dir(output_dir / "data")
    ensure_dir(output_dir / "videos")

    episode_dirs = sorted(
        [p for p in input_dir.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    if not episode_dirs:
        raise FileNotFoundError(f"No raw episodes found under '{input_dir}'")

    want_images = dataset_has_any_rgb(input_dir, episode_dirs)
    if not want_images:
        raise ValueError(
            f"No frame.png files found anywhere under '{input_dir}'. This does not look "
            f"like a vla/unified capture (it may be hitl_d or hitl_hgd, which record no "
            f"RGB). This converter targets VLA training and refuses to emit an "
            f"'observation.images.cam_high' feature with no actual video data. If you "
            f"intended a non-image conversion, this script is not the right tool."
        )

    if last_frame_action not in ("drop", "repeat"):
        raise ValueError("--last-frame-action must be 'drop' or 'repeat'")

    chunk_name = "chunk-000"
    output_chunk_dir = output_dir / "data" / chunk_name
    ensure_dir(output_chunk_dir)

    episode_rows: List[Dict[str, Any]] = []
    task_text_to_index: Dict[str, int] = {}
    task_rows: List[Dict[str, Any]] = []
    all_state_vectors: List[np.ndarray] = []
    all_action_vectors: List[np.ndarray] = []
    episodes_skipped_empty_task = 0

    for episode_index, episode_dir in enumerate(episode_dirs):
        episode_id = f"episode_{episode_index:06d}"
        timestep_dirs = sorted(
            [p for p in episode_dir.iterdir() if p.is_dir() and p.name.isdigit()],
            key=lambda p: int(p.name),
        )
        if not timestep_dirs:
            continue

        # NOTE: record.py already subsamples to <=512 frames at record time via
        # select_evenly_spaced; this re-subsampling only kicks in if timestep_dirs
        # somehow exceeds that (e.g. data from an older/different recorder config).
        if len(timestep_dirs) > MAX_FRAMES_PER_EPISODE:
            timestep_dirs = [
                timestep_dirs[int(i)]
                for i in np.linspace(0, len(timestep_dirs) - 1, MAX_FRAMES_PER_EPISODE, dtype=int)
            ]

        # Pass 1: load every timestep that has BOTH low_dim.npy and frame.png.
        # A timestep missing either is dropped entirely, keeping state/frame in lockstep --
        # we never advance one list without the other.
        state_vecs: List[np.ndarray] = []
        ee_poses: List[np.ndarray] = []
        frame_paths: List[Path] = []
        for timestep_dir in timestep_dirs:
            low_dim_path = timestep_dir / "low_dim.npy"
            frame_path = timestep_dir / "frame.png"
            if not low_dim_path.exists() or not frame_path.exists():
                continue
            state_info = np.load(low_dim_path, allow_pickle=True).item()
            state_vecs.append(flatten_robot_state(state_info))
            ee_poses.append(extract_ee_pose(state_info))
            frame_paths.append(frame_path)

        n = len(state_vecs)
        if n < 2:
            # Need at least 2 valid timesteps to derive a next-state action for frame 0.
            continue

        # Pass 2: build action[i] = ee_pose[i+1] (next observed pose). The last frame
        # has no "next" -- either drop it (default) or repeat the previous action.
        task_value = resolve_task_for_episode(episode_dir.name, episode_index, tasks_map, global_task)
        if not task_value and not allow_empty_task:
            episodes_skipped_empty_task += 1
            continue

        rows: List[Dict[str, Any]] = []
        used_frames: List[Path] = []
        last_i = n if last_frame_action == "repeat" else n - 1
        for i in range(last_i):
            action_source_i = min(i + 1, n - 1)
            action_vec = ee_poses[action_source_i]
            state_vec = state_vecs[i]
            all_state_vectors.append(state_vec)
            all_action_vectors.append(action_vec)
            rows.append({
                "observation.state": state_vec.astype(np.float32).tolist(),
                "action": action_vec.astype(np.float32).tolist(),
                "task": task_value,
                "frame_index": i,
            })
            used_frames.append(frame_paths[i])

        if not rows:
            continue

        assert len(rows) == len(used_frames), (
            f"{episode_id}: row/frame count mismatch ({len(rows)} vs {len(used_frames)}) "
            f"-- refusing to write a misaligned episode."
        )

        parquet_path = output_chunk_dir / f"{episode_id}.parquet"
        pd.DataFrame(rows).to_parquet(parquet_path, index=False)

        video_dir = output_dir / "videos" / chunk_name / "observation.images.cam_high"
        video_path = video_dir / f"{episode_id}.mp4"
        n_written = write_video_from_frames(used_frames, video_path, fps=fps)
        assert n_written == len(rows), (
            f"{episode_id}: wrote {n_written} video frames but have {len(rows)} parquet "
            f"rows -- aborting before producing a misaligned episode."
        )

        if task_value not in task_text_to_index:
            task_text_to_index[task_value] = len(task_text_to_index)
            task_rows.append({
                "task_index": task_text_to_index[task_value],
                "task": task_value,
            })

        episode_rows.append({
            "episode_id": episode_id,
            "chunk": chunk_name,
            "file": str(parquet_path.relative_to(output_dir)),
            "video_file": str(video_path.relative_to(output_dir)),
            "num_frames": len(rows),
            "task": task_value,
            "task_index": task_text_to_index[task_value],
            "source_episode": episode_dir.name,
            "fps": fps,
            "fps_is_nominal": True,
            "robot_type": "kinova_gen3",
        })

    if episodes_skipped_empty_task:
        print(
            f"WARNING: skipped {episodes_skipped_empty_task} episode(s) with no resolvable "
            f"task text. Pass --task, --tasks-file, or --allow-empty-task to include them.",
            file=sys.stderr,
        )

    if not episode_rows:
        raise FileNotFoundError(
            f"No valid episodes produced from '{input_dir}'. Check that episodes have "
            f">=2 timesteps with both low_dim.npy and frame.png, and that a task could "
            f"be resolved for each (see --task / --tasks-file / --allow-empty-task)."
        )

    # --- stats ---
    state_matrix = np.vstack([v for v in all_state_vectors if v.size > 0])
    action_matrix = np.vstack([v for v in all_action_vectors if v.size > 0])
    state_mean = state_matrix.mean(axis=0).astype(np.float32)
    state_std = np.where(state_matrix.std(axis=0) < 1e-8, 1.0, state_matrix.std(axis=0)).astype(np.float32)
    action_mean = action_matrix.mean(axis=0).astype(np.float32)
    action_std = np.where(action_matrix.std(axis=0) < 1e-8, 1.0, action_matrix.std(axis=0)).astype(np.float32)

    write_stats_safetensors(output_meta_dir / "stats.safetensors", {
        "observation.state_mean": state_mean,
        "observation.state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std,
    })

    with (output_meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for record in episode_rows:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with (output_meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for record in task_rows:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    info = {
        "format_version": "lerobot_v2",
        "dataset_name": repo_id or input_dir.name,
        "source_dataset": str(input_dir),
        "robot_type": "kinova_gen3",
        "fps": fps,
        "fps_is_nominal": True,
        "action_definition": (
            "next observed end-effector pose (position + orientation) at t+1; "
            "NOT a recorded commanded control signal"
        ),
        "last_frame_action_policy": last_frame_action,
        "features": {
            "observation.state": "vector",
            "action": "vector",
            "observation.images.cam_high": "video",
            "task": "text",
        },
        "episode_count": len(episode_rows),
        "chunk_count": 1,
    }
    (output_meta_dir / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Converted raw unified dataset {input_dir} -> {output_dir}")
    print(f"Episodes written: {len(episode_rows)}")
    print(f"Distinct tasks: {len(task_rows)}")


# --------------------------------------------------------------------------
# already-lerobot-formatted dataset -> re-normalized lerobot v2
# (passthrough path: assumes source already has real, distinct action data)
# --------------------------------------------------------------------------

def discover_input_files(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        return []
    return sorted(data_dir.rglob("*.parquet"))


def build_episode_table(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return df
    renamed = {}
    for column in list(df.columns):
        low = column.lower()
        if low in {"state", "observation.state", "proprio", "proprioception", "robot_state"}:
            renamed[column] = "observation.state"
        elif low in {"action", "actions", "control", "command"}:
            renamed[column] = "action"
        elif low in {"task", "instruction", "language_instruction", "prompt", "text"}:
            renamed[column] = "task"
    if renamed:
        df = df.rename(columns=renamed)
    return df


def infer_episode_id_from_dataframe(df: pd.DataFrame, fallback: str) -> str:
    for candidate in ["episode_id", "episode", "episode_index", "id"]:
        if candidate in df.columns:
            values = df[candidate].dropna()
            if len(values):
                return str(values.iloc[0])
    return fallback


def convert_prelabeled_lerobot_dataset(input_dir: Path, output_dir: Path, repo_id: Optional[str]) -> None:
    meta_dir = input_dir / "meta"
    source_meta = load_json(meta_dir / "info.json") if (meta_dir / "info.json").exists() else {}
    source_episodes = load_jsonl(meta_dir / "episodes.jsonl")
    source_tasks = load_jsonl(meta_dir / "tasks.jsonl")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_meta_dir = output_dir / "meta"
    ensure_dir(output_meta_dir)
    ensure_dir(output_dir / "data")

    parquet_files = discover_input_files(input_dir / "data")
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under '{input_dir / 'data'}'")

    source_task_map: Dict[str, Any] = {
        str(row["episode_id"]): row for row in source_tasks if "episode_id" in row
    }

    episode_stats: Dict[str, np.ndarray] = {}
    episode_records: List[Dict[str, Any]] = []
    task_records: List[Dict[str, Any]] = []

    for parquet_path in parquet_files:
        rel = parquet_path.relative_to(input_dir / "data")
        chunk_name = rel.parts[0] if len(rel.parts) > 1 else "chunk-000"
        episode_name = parquet_path.stem
        output_parquet = output_dir / "data" / chunk_name / f"{episode_name}.parquet"
        ensure_dir(output_parquet.parent)

        df = build_episode_table(parquet_path)
        if df.empty:
            continue

        episode_id = infer_episode_id_from_dataframe(df, episode_name)

        for col_name, alt in (("observation.state", "state"), ("action", "actions")):
            if col_name not in df.columns and alt in df.columns:
                df = df.rename(columns={alt: col_name})
            if col_name in df.columns:
                df[col_name] = df[col_name].apply(
                    lambda x: np.asarray(x, dtype=np.float32).reshape(-1).tolist() if x is not None else []
                )

        if "task" not in df.columns:
            task_value = ""
            if episode_id in source_task_map:
                task_value = find_task_value(source_task_map[episode_id])
            elif source_tasks:
                task_value = find_task_value(source_tasks[0])
            if task_value:
                df["task"] = task_value

        df.to_parquet(output_parquet, index=False)

        for key, value in collect_stats_from_episode(df).items():
            episode_stats[key] = value if key not in episode_stats else np.maximum(episode_stats[key], value)

        episode_record = {
            "episode_id": episode_id,
            "chunk": chunk_name,
            "file": str(output_parquet.relative_to(output_dir)),
            "num_frames": int(len(df)),
            "task": df["task"].iloc[0] if "task" in df.columns and not df["task"].empty else "",
        }
        if source_episodes and len(source_episodes) >= len(episode_records) + 1:
            episode_record.update(source_episodes[len(episode_records)])
        else:
            for key in ["fps", "robot_type"]:
                if key in source_meta:
                    episode_record[key] = source_meta[key]
        episode_records.append(episode_record)

        task_value = episode_record.get("task", "")
        if not task_value:
            source_task = source_task_map.get(episode_id)
            if source_task is not None:
                task_value = find_task_value(source_task)
        task_records.append({
            "episode_id": episode_id,
            "task": task_value,
            "language": source_meta.get("language", "en"),
        })

    if not episode_records:
        raise FileNotFoundError(f"No valid episodes found in '{input_dir}'")

    combined_stats: Dict[str, np.ndarray] = {}
    for key, value in episode_stats.items():
        if key.endswith("_mean") and f"{key[:-5]}_std" in episode_stats:
            combined_stats[key] = value
            combined_stats[f"{key[:-5]}_std"] = episode_stats[f"{key[:-5]}_std"]
    if not combined_stats:
        combined_stats = {
            "observation.state_mean": np.zeros(1, dtype=np.float32),
            "observation.state_std": np.ones(1, dtype=np.float32),
            "action_mean": np.zeros(1, dtype=np.float32),
            "action_std": np.ones(1, dtype=np.float32),
        }
    write_stats_safetensors(output_meta_dir / "stats.safetensors", combined_stats)

    with (output_meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for record in episode_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output_meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for record in task_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    info = dict(source_meta)
    info.update({
        "format_version": "lerobot_v2",
        "dataset_name": repo_id or output_dir.name,
        "source_dataset": str(input_dir),
        "robot_type": info.get("robot_type", "custom"),
        "fps": info.get("fps", 30),
        "features": {
            "observation.state": "vector",
            "action": "vector",
            "observation.images.cam_high": "video",
            "observation.images.cam_wrist": "video",
            "task": "text",
        },
        "episode_count": len(episode_records),
        "chunk_count": len({ep.get("chunk") for ep in episode_records}),
    })
    (output_meta_dir / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    copy_video_tree(input_dir, output_dir)

    print(f"Converted {input_dir} -> {output_dir}")
    print(f"Episodes written: {len(episode_records)}")
    print(f"Task rows written: {len(task_records)}")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def convert_dataset(
    input_dir: Path,
    output_dir: Path,
    repo_id: Optional[str],
    fps: float,
    global_task: Optional[str],
    tasks_map: Dict[str, str],
    allow_empty_task: bool,
    last_frame_action: str,
) -> None:
    if detect_raw_unified_dataset(input_dir):
        convert_raw_unified_dataset(
            input_dir, output_dir, repo_id=repo_id, fps=fps,
            global_task=global_task, tasks_map=tasks_map,
            allow_empty_task=allow_empty_task, last_frame_action=last_frame_action,
        )
        return

    if not (input_dir / "meta").exists():
        raise FileNotFoundError(
            f"Input dataset '{input_dir}' must either be a raw unified capture "
            f"(numeric episode folders) or already contain a meta/ directory"
        )
    convert_prelabeled_lerobot_dataset(input_dir, output_dir, repo_id=repo_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a unified dataset to LeRobot v2 for VLA pi0 workloads.")
    parser.add_argument("input_dataset", type=Path, help="Path to the unified dataset root")
    parser.add_argument("output_dataset", type=Path, help="Directory to write the LeRobot v2 dataset to")
    parser.add_argument("--repo-id", default=None, help="Optional dataset repo id used in meta/info.json")
    parser.add_argument("--fps", type=float, default=30.0, help="Nominal fps written to info.json (NOT measured -- see module docstring)")
    parser.add_argument("--task", default=None, help="Single language instruction applied to every episode without a more specific mapping")
    parser.add_argument("--tasks-file", type=Path, default=None, help="JSON file mapping raw episode dir name (e.g. '0') to task text")
    parser.add_argument("--allow-empty-task", action="store_true", help="Allow episodes with no resolvable task text to be written with task=''")
    parser.add_argument("--last-frame-action", choices=["drop", "repeat"], default="drop", help="What to do with the final frame, which has no 'next' pose to use as its action target")
    args = parser.parse_args()

    tasks_map: Dict[str, str] = {}
    if args.tasks_file is not None:
        raw_map = load_json(args.tasks_file)
        if not isinstance(raw_map, dict):
            raise SystemExit("--tasks-file must contain a JSON object mapping episode name/index -> task text")
        tasks_map = {str(k): str(v) for k, v in raw_map.items()}

    convert_dataset(
        args.input_dataset, args.output_dataset, repo_id=args.repo_id, fps=args.fps,
        global_task=args.task, tasks_map=tasks_map,
        allow_empty_task=args.allow_empty_task, last_frame_action=args.last_frame_action,
    )


if __name__ == "__main__":
    main()