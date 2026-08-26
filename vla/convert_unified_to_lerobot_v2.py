#!/usr/bin/env python3
"""Convert a unified dataset into a LeRobot v2 dataset layout for pi0-style VLA training.

Expected input layout:
    <dataset_repo_id>/
    ├── meta/
    │   ├── info.json
    │   ├── episodes.jsonl
    │   ├── tasks.jsonl
    │   └── stats.safetensors  # optional
    ├── data/
    │   └── chunk-000/
    │       ├── episode_000000.parquet
    │       └── episode_000001.parquet
    └── videos/
        └── chunk-000/
            ├── observation.images.cam_high/
            │   └── episode_000000.mp4
            └── observation.images.cam_wrist/
                └── episode_000000.mp4

The output mirrors the LeRobot v2 dataset layout used by pi0 and other VLA pipelines:
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
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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


def episode_id_from_path(path: Path) -> str:
    if path.stem.startswith("episode_"):
        return path.stem
    return f"episode_{path.stem:06d}"


def to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray,)):
        return value.astype(float).tolist()
    if isinstance(value, (list, tuple)):
        return [to_python_scalar(v) for v in value]
    return value


def canonicalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in record.items():
        if key in {"state", "proprio", "proprioception", "robot_state"}:
            out["observation.state"] = to_python_scalar(value)
        elif key in {"action", "actions", "control", "command"}:
            out["action"] = to_python_scalar(value)
        elif key.startswith("image") or key.startswith("images") or key in {"cam_high", "cam_wrist", "front", "wrist"}:
            camera_name = key.replace("image_", "").replace("images_", "").replace("_", ".")
            out[f"observation.images.{camera_name}"] = to_python_scalar(value)
        elif key in {"task", "instruction", "language_instruction", "prompt", "text"}:
            out["task"] = str(value)
        else:
            out[key] = to_python_scalar(value)
    return out


def infer_episode_id_from_dataframe(df: pd.DataFrame, fallback: str) -> str:
    for candidate in ["episode_id", "episode", "episode_index", "id"]:
        if candidate in df.columns:
            values = df[candidate].dropna()
            if len(values):
                return str(values.iloc[0])
    return fallback


def find_task_value(record: Dict[str, Any]) -> str:
    for key in DEFAULT_TASK_KEYS:
        if key in record and record[key] not in (None, ""):
            return str(record[key])
    return ""


def coerce_vector_column(series: pd.Series) -> np.ndarray:
    vecs = []
    for value in series.tolist():
        if value is None:
            vecs.append(np.zeros(0, dtype=np.float32))
            continue
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            vecs.append(np.asarray([float(arr)], dtype=np.float32))
        else:
            vecs.append(arr.astype(np.float32).reshape(-1))
    return np.asarray(vecs, dtype=np.float32)


def collect_stats_from_episode(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    stats: Dict[str, np.ndarray] = {}
    for column in ["observation.state", "action"]:
        if column in df.columns:
            arr = coerce_vector_column(df[column]).astype(np.float32)
            if arr.size > 0:
                stats[f"{column}_mean"] = arr.mean(axis=0)
                stats[f"{column}_std"] = arr.std(axis=0)
                stats[f"{column}_std"] = np.where(stats[f"{column}_std"] < 1e-8, 1.0, stats[f"{column}_std"])
    return stats


def write_stats_safetensors(path: Path, stats_by_key: Dict[str, np.ndarray]) -> None:
    if save_safetensors_file is None:
        raise RuntimeError("safetensors is not installed. Install it with: pip install safetensors")
    tensor_dict = {key: np.asarray(value, dtype=np.float32) for key, value in stats_by_key.items()}
    save_safetensors_file(tensor_dict, path)


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


def write_video_from_frames(frame_paths: List[Path], output_mp4: Path) -> None:
    if not frame_paths:
        return
    import cv2  # import here to avoid conflict with other libraries
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        return
    height, width, _ = first.shape
    fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_mp4), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_mp4}")
    for path in frame_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        writer.write(frame)
    writer.release()


def flatten_robot_state(state_info: Dict[str, Any]) -> np.ndarray:
    values: List[float] = []
    joints = state_info.get("joints", {})
    pos = joints.get("position", [])
    if isinstance(pos, (list, tuple, np.ndarray)):
        values.extend(np.asarray(pos, dtype=np.float32).reshape(-1).tolist())
    ee_pos = state_info.get("ee_position", [])
    if isinstance(ee_pos, (list, tuple, np.ndarray)):
        values.extend(np.asarray(ee_pos, dtype=np.float32).reshape(-1).tolist())
    ee_ori = state_info.get("ee_orientation", [])
    if isinstance(ee_ori, (list, tuple, np.ndarray)):
        arr = np.asarray(ee_ori, dtype=np.float32).reshape(-1)
        values.extend(arr.tolist())
    centroids = state_info.get("centroids", [])
    if isinstance(centroids, (list, tuple, np.ndarray)):
        values.extend(np.asarray(centroids, dtype=np.float32).reshape(-1).tolist())
    if not values:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def flatten_action(state_info: Dict[str, Any]) -> np.ndarray:
    action_values: List[float] = []
    ee_pos = state_info.get("ee_position", [])
    if isinstance(ee_pos, (list, tuple, np.ndarray)):
        action_values.extend(np.asarray(ee_pos, dtype=np.float32).reshape(-1).tolist())
    ee_ori = state_info.get("ee_orientation", [])
    if isinstance(ee_ori, (list, tuple, np.ndarray)):
        action_values.extend(np.asarray(ee_ori, dtype=np.float32).reshape(-1).tolist())
    if not action_values:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(action_values, dtype=np.float32)


def detect_raw_unified_dataset(input_dir: Path) -> bool:
    if (input_dir / "meta").exists():
        return False
    if not input_dir.exists():
        return False
    children = [p for p in input_dir.iterdir() if p.is_dir()]
    for child in children:
        if child.name.isdigit():
            return True
    return False


def convert_raw_unified_dataset(input_dir: Path, output_dir: Path, repo_id: str | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_meta_dir = output_dir / "meta"
    ensure_dir(output_meta_dir)
    ensure_dir(output_dir / "data")
    ensure_dir(output_dir / "videos")

    episode_rows: List[Dict[str, Any]] = []
    task_rows: List[Dict[str, Any]] = []
    all_state_vectors: List[np.ndarray] = []
    all_action_vectors: List[np.ndarray] = []

    episode_dirs = sorted(
        [p for p in input_dir.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    if not episode_dirs:
        raise FileNotFoundError(f"No raw episodes found under '{input_dir}'")

    chunk_name = "chunk-000"
    output_chunk_dir = output_dir / "data" / chunk_name
    ensure_dir(output_chunk_dir)

    for episode_index, episode_dir in enumerate(episode_dirs):
        episode_id = f"episode_{episode_index:06d}"
        timestep_dirs = sorted(
            [p for p in episode_dir.iterdir() if p.is_dir() and p.name.isdigit()],
            key=lambda p: int(p.name),
        )
        if not timestep_dirs:
            continue

        selected_indices = timestep_dirs
        if len(timestep_dirs) > 512:
            selected_indices = [timestep_dirs[int(i)] for i in np.linspace(0, len(timestep_dirs) - 1, 512, dtype=int)]

        frames: List[Path] = []
        rows: List[Dict[str, Any]] = []
        for timestep_dir in selected_indices:
            low_dim_path = timestep_dir / "low_dim.npy"
            if not low_dim_path.exists():
                continue
            state_info = np.load(low_dim_path, allow_pickle=True).item()
            state_vec = flatten_robot_state(state_info)
            action_vec = flatten_action(state_info)
            if state_vec.size == 0:
                state_vec = np.asarray([], dtype=np.float32)
            if action_vec.size == 0:
                action_vec = np.asarray([], dtype=np.float32)
            all_state_vectors.append(state_vec)
            all_action_vectors.append(action_vec)
            rows.append({
                "observation.state": state_vec.astype(np.float32).tolist(),
                "action": action_vec.astype(np.float32).tolist(),
                "task": "",
            })

            frame_path = timestep_dir / "frame.png"
            if frame_path.exists():
                frames.append(frame_path)

        if not rows:
            continue

        parquet_path = output_chunk_dir / f"{episode_id}.parquet"
        pd.DataFrame(rows).to_parquet(parquet_path, index=False)

        episode_rows.append({
            "episode_id": episode_id,
            "chunk": chunk_name,
            "file": str(parquet_path.relative_to(output_dir)),
            "num_frames": len(rows),
            "source_episode": str(episode_dir.name),
            "fps": 30,
            "robot_type": "kinova_gen3",
        })
        task_rows.append({
            "episode_id": episode_id,
            "task": "",
            "language": "en",
        })

        if frames:
            video_dir = output_dir / "videos" / chunk_name / "observation.images.cam_high"
            ensure_dir(video_dir)
            write_video_from_frames(frames, video_dir / f"{episode_id}.mp4")

    if not episode_rows:
        raise FileNotFoundError(f"No valid episodes found in '{input_dir}'")

    if all_state_vectors:
        state_matrix = np.vstack([v for v in all_state_vectors if v.size > 0])
        action_matrix = np.vstack([v for v in all_action_vectors if v.size > 0])
        state_mean = state_matrix.mean(axis=0).astype(np.float32)
        state_std = state_matrix.std(axis=0).astype(np.float32)
        state_std = np.where(state_std < 1e-8, 1.0, state_std)
        action_mean = action_matrix.mean(axis=0).astype(np.float32)
        action_std = action_matrix.std(axis=0).astype(np.float32)
        action_std = np.where(action_std < 1e-8, 1.0, action_std)
    else:
        state_mean = np.zeros(1, dtype=np.float32)
        state_std = np.ones(1, dtype=np.float32)
        action_mean = np.zeros(1, dtype=np.float32)
        action_std = np.ones(1, dtype=np.float32)

    stats = {
        "observation.state_mean": state_mean,
        "observation.state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }
    write_stats_safetensors(output_meta_dir / "stats.safetensors", stats)

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
        "fps": 30,
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


def convert_dataset(input_dir: Path, output_dir: Path, repo_id: str | None = None) -> None:
    if detect_raw_unified_dataset(input_dir):
        convert_raw_unified_dataset(input_dir, output_dir, repo_id=repo_id)
        return

    if not (input_dir / "meta").exists():
        raise FileNotFoundError(f"Input dataset '{input_dir}' must either be a raw unified dataset or contain a meta/ directory")

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

    episode_stats: Dict[str, np.ndarray] = {}
    episode_records: List[Dict[str, Any]] = []
    task_records: List[Dict[str, Any]] = []
    task_lookup: Dict[str, str] = {}
    source_task_map: Dict[str, Any] = {}

    for row in source_tasks:
        if "episode_id" in row:
            source_task_map[str(row["episode_id"])] = row

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
        if "observation.state" not in df.columns and "state" in df.columns:
            df = df.rename(columns={"state": "observation.state"})
        if "action" not in df.columns and "actions" in df.columns:
            df = df.rename(columns={"actions": "action"})

        if "observation.state" in df.columns:
            df["observation.state"] = df["observation.state"].apply(lambda x: np.asarray(x, dtype=np.float32).reshape(-1).tolist() if x is not None else [])
        if "action" in df.columns:
            df["action"] = df["action"].apply(lambda x: np.asarray(x, dtype=np.float32).reshape(-1).tolist() if x is not None else [])

        if "task" not in df.columns:
            task_value = ""
            if episode_id in source_task_map:
                task_value = find_task_value(source_task_map[episode_id])
            elif source_tasks:
                task_value = find_task_value(source_tasks[0]) if source_tasks else ""
            if task_value:
                df["task"] = task_value

        df.to_parquet(output_parquet, index=False)

        episode_stats_for_file = collect_stats_from_episode(df)
        for key, value in episode_stats_for_file.items():
            if key not in episode_stats:
                episode_stats[key] = value
            else:
                episode_stats[key] = np.maximum(episode_stats[key], value)

        episode_record = {
            "episode_id": episode_id,
            "chunk": chunk_name,
            "file": str(output_parquet.relative_to(output_dir)),
            "num_frames": int(len(df)),
            "task": df["task"].iloc[0] if "task" in df.columns and not df["task"].empty else "",
        }
        if source_episodes and len(source_episodes) >= len(episode_records) + 1:
            source_record = source_episodes[len(episode_records)]
            episode_record.update(source_record)
        else:
            for key in ["fps", "robot_type"]:
                if key in source_meta:
                    episode_record[key] = source_meta[key]

        episode_records.append(episode_record)

        task_value = episode_record.get("task", "")
        if task_value:
            task_lookup[episode_id] = task_value
            task_records.append({
                "episode_id": episode_id,
                "task": task_value,
                "language": source_meta.get("language", "en"),
            })
        else:
            source_task = source_task_map.get(episode_id)
            if source_task is not None:
                task_value = find_task_value(source_task)
                if task_value:
                    task_lookup[episode_id] = task_value
                    task_records.append({
                        "episode_id": episode_id,
                        "task": task_value,
                        "language": source_meta.get("language", "en"),
                    })

    if not task_records:
        for episode in episode_records:
            ep_id = str(episode.get("episode_id", "episode_000000"))
            task_records.append({
                "episode_id": ep_id,
                "task": "",
                "language": source_meta.get("language", "en"),
            })

    combined_stats: Dict[str, np.ndarray] = {}
    for key, value in episode_stats.items():
        if key.endswith("_mean"):
            base = key[:-5]
            other = episode_stats.get(f"{base}_std")
            if other is not None:
                combined_stats[key] = value
                combined_stats[f"{base}_std"] = other
        elif key.endswith("_std") and key[:-4] + "_mean" not in episode_stats:
            combined_stats[key] = value

    if not combined_stats:
        # Fallback stats: all-zero mean, unit std. This keeps the dataset valid even when
        # the source parquet files do not contain a state/action vector column.
        combined_stats = {"observation.state_mean": np.zeros(1, dtype=np.float32), "observation.state_std": np.ones(1, dtype=np.float32), "action_mean": np.zeros(1, dtype=np.float32), "action_std": np.ones(1, dtype=np.float32)}

    stats_path = output_meta_dir / "stats.safetensors"
    write_stats_safetensors(stats_path, combined_stats)

    if source_episodes:
        episode_file = output_meta_dir / "episodes.jsonl"
        with episode_file.open("w", encoding="utf-8") as f:
            for record in episode_records or source_episodes:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        episode_file = output_meta_dir / "episodes.jsonl"
        with episode_file.open("w", encoding="utf-8") as f:
            for record in episode_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    task_file = output_meta_dir / "tasks.jsonl"
    with task_file.open("w", encoding="utf-8") as f:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a unified dataset to LeRobot v2 for VLA pi0 workloads.")
    parser.add_argument("input_dataset", type=Path, help="Path to the unified dataset root (contains meta/, data/, videos/)")
    parser.add_argument("output_dataset", type=Path, help="Directory to write the LeRobot v2 dataset to")
    parser.add_argument("--repo-id", default=None, help="Optional dataset repo id used in meta/info.json")
    args = parser.parse_args()

    convert_dataset(args.input_dataset, args.output_dataset, repo_id=args.repo_id)


if __name__ == "__main__":
    main()
