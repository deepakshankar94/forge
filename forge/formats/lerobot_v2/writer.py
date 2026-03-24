"""LeRobot v2 format writer for Forge.

Writes datasets in legacy LeRobot v2.x layout:
    dataset/
    ├── meta/
    │   ├── info.json
    │   ├── tasks.jsonl
    │   ├── episodes.jsonl
    │   └── episodes_stats.jsonl
    ├── data/
    │   └── chunk-000/
    │       ├── episode_000000.parquet
    │       └── ...
    └── videos/
        └── chunk-000/
            ├── observation.images.camera0/
            │   ├── episode_000000.mp4
            │   └── ...
            └── ...
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.exceptions import ConversionError, MissingDependencyError
from forge.core.models import CameraInfo, DatasetInfo, Episode, LazyImage
from forge.formats.registry import FormatRegistry
from forge.video.encoder import VideoEncoder, VideoEncoderConfig

LEGACY_CODEBASE_VERSION = "v2.1"
LEGACY_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
LEGACY_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

STANDARD_FEATURES: dict[str, dict[str, Any]] = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}


def _check_pyarrow() -> None:
    """Check if PyArrow is available."""
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise MissingDependencyError(
            dependency="pyarrow",
            feature="LeRobot v2 format writing",
            install_hint="pip install forge-robotics[lerobot]",
        )


def _infer_dtype(value: Any) -> str:
    """Infer a LeRobot-compatible dtype string from a numpy-compatible value."""
    arr = np.asarray(value)
    dtype = arr.dtype

    if np.issubdtype(dtype, np.bool_):
        return "bool"
    if np.issubdtype(dtype, np.unsignedinteger):
        return "uint8" if dtype.itemsize <= 1 else "int64"
    if np.issubdtype(dtype, np.integer):
        return "int32" if dtype.itemsize <= 4 else "int64"
    if np.issubdtype(dtype, np.floating):
        return "float64" if dtype.itemsize > 4 else "float32"
    return "float32"


class ArrayStatsAccumulator:
    """Running stats accumulator for numeric scalars and arrays."""

    def __init__(self) -> None:
        self.count = 0
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None
        self.minimum: np.ndarray | None = None
        self.maximum: np.ndarray | None = None

    def update(self, value: Any) -> None:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            arr = arr.reshape(1)

        if self.mean is None:
            self.mean = arr.copy()
            self.m2 = np.zeros_like(arr, dtype=np.float64)
            self.minimum = arr.copy()
            self.maximum = arr.copy()
            self.count = 1
            return

        self.count += 1
        delta = arr - self.mean
        self.mean += delta / self.count
        delta2 = arr - self.mean
        self.m2 += delta * delta2
        self.minimum = np.minimum(self.minimum, arr)
        self.maximum = np.maximum(self.maximum, arr)

    def to_serializable(self) -> dict[str, Any]:
        if self.mean is None or self.m2 is None or self.minimum is None or self.maximum is None:
            return {}

        std = np.sqrt(self.m2 / max(self.count, 1))
        return {
            "min": self.minimum.astype(np.float32).tolist(),
            "max": self.maximum.astype(np.float32).tolist(),
            "mean": self.mean.astype(np.float32).tolist(),
            "std": std.astype(np.float32).tolist(),
            "count": [int(self.count)],
        }


class ImageStatsAccumulator:
    """Running per-channel stats accumulator for RGB video frames."""

    def __init__(self) -> None:
        self.frame_count = 0
        self.pixel_count = 0
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None
        self.minimum: np.ndarray | None = None
        self.maximum: np.ndarray | None = None

    def update(self, image: Any) -> None:
        arr = np.asarray(image)
        if arr.ndim != 3:
            raise ValueError(f"Expected HWC image array, got shape {arr.shape}")

        if np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)
            if arr.size > 0 and arr.max() > 1.0:
                arr = np.clip(arr, 0.0, 255.0) / 255.0
        else:
            arr = np.clip(arr.astype(np.float32), 0.0, 255.0) / 255.0

        pixels = arr.reshape(-1, arr.shape[-1])
        batch_count = pixels.shape[0]
        batch_mean = pixels.mean(axis=0, dtype=np.float64)
        batch_var = pixels.var(axis=0, dtype=np.float64)
        batch_min = pixels.min(axis=0)
        batch_max = pixels.max(axis=0)

        if self.mean is None:
            self.mean = batch_mean
            self.m2 = batch_var * batch_count
            self.minimum = batch_min
            self.maximum = batch_max
            self.pixel_count = batch_count
            self.frame_count = 1
            return

        total_count = self.pixel_count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / total_count)
        self.m2 = (
            self.m2
            + (batch_var * batch_count)
            + (delta**2) * self.pixel_count * batch_count / total_count
        )
        self.minimum = np.minimum(self.minimum, batch_min)
        self.maximum = np.maximum(self.maximum, batch_max)
        self.pixel_count = total_count
        self.frame_count += 1

    def to_serializable(self) -> dict[str, Any]:
        if (
            self.mean is None
            or self.m2 is None
            or self.minimum is None
            or self.maximum is None
            or self.pixel_count == 0
        ):
            return {}

        std = np.sqrt(self.m2 / self.pixel_count)
        reshape = lambda x: np.asarray(x, dtype=np.float32).reshape(-1, 1, 1).tolist()
        return {
            "min": reshape(self.minimum),
            "max": reshape(self.maximum),
            "mean": reshape(self.mean),
            "std": reshape(std),
            "count": [int(self.frame_count)],
        }


@dataclass
class LeRobotV2WriterConfig:
    """Configuration for LeRobot v2 writer."""

    fps: float = 30.0
    robot_type: str = "unknown"
    video_codec: str = "libx264"
    video_crf: int = 23
    video_preset: str = "medium"
    chunks_size: int = 1000
    repo_id: str | None = None
    camera_name_mapping: dict[str, str] = field(default_factory=dict)


@FormatRegistry.register_writer("lerobot-v2")
class LeRobotV2Writer:
    """Writer for legacy LeRobot v2.x datasets."""

    def __init__(self, config: LeRobotV2WriterConfig | None = None):
        self.config = config or LeRobotV2WriterConfig()
        self._video_encoder = VideoEncoder(
            VideoEncoderConfig(
                codec=self.config.video_codec,
                crf=self.config.video_crf,
                preset=self.config.video_preset,
            )
        )
        self._reset_state()

    @property
    def format_name(self) -> str:
        return "lerobot-v2"

    def _reset_state(self) -> None:
        self._episode_metadata: list[dict[str, Any]] = []
        self._episode_stats: list[dict[str, Any]] = []
        self._task_metadata: list[dict[str, Any]] = []
        self._tasks_seen: dict[str, int] = {}
        self._task_index_to_task: dict[int, str] = {}
        self._cameras: dict[str, CameraInfo] = {}
        self._features: dict[str, dict[str, Any]] = {
            key: value.copy() for key, value in STANDARD_FEATURES.items()
        }
        self._total_frames = 0
        self._total_videos = 0

    def _map_camera_name(self, source_name: str) -> str:
        """Map source camera name to LeRobot naming convention."""
        if source_name in self.config.camera_name_mapping:
            return self.config.camera_name_mapping[source_name]

        clean_name = source_name
        if clean_name.endswith("_image"):
            clean_name = clean_name[:-6]
        if clean_name.endswith("_rgb"):
            clean_name = clean_name[:-4]

        return f"observation.images.{clean_name}"

    def _resolve_task(self, episode: Episode) -> str:
        """Resolve the task string for an episode."""
        return (
            episode.language_instruction
            or episode.metadata.get("task")
            or episode.metadata.get("language_instruction")
            or "default"
        )

    def _get_or_create_task_index(self, task: str, preferred_index: int | None = None) -> int:
        """Get or create a stable task index, preserving source indices when provided."""
        if task in self._tasks_seen:
            return self._tasks_seen[task]

        task_index = preferred_index
        if task_index is None or task_index in self._task_index_to_task:
            task_index = 0
            if self._task_index_to_task:
                task_index = max(self._task_index_to_task) + 1

        self._tasks_seen[task] = task_index
        self._task_index_to_task[task_index] = task
        self._task_metadata.append({"task_index": task_index, "task": task})
        self._task_metadata.sort(key=lambda item: int(item["task_index"]))
        return task_index

    def _ensure_array_feature(self, name: str, value: Any) -> None:
        """Track array feature metadata for info.json."""
        if name in self._features:
            return

        arr = np.asarray(value)
        self._features[name] = {
            "dtype": _infer_dtype(arr),
            "shape": list(arr.shape),
            "names": None,
        }

    def _ensure_video_feature(self, name: str, image: LazyImage, fps: float) -> None:
        """Track video feature metadata for info.json."""
        if name in self._features:
            return

        self._features[name] = {
            "dtype": "video",
            "shape": [image.height, image.width, image.channels],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": float(fps),
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }

    def _episode_chunk(self, episode_index: int) -> int:
        return episode_index // self.config.chunks_size

    def write_episode(
        self,
        episode: Episode,
        output_path: Path,
        episode_index: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """Write a single episode in legacy LeRobot v2 layout."""
        _check_pyarrow()
        import pyarrow as pa
        import pyarrow.parquet as pq

        if episode_index is None:
            episode_index = len(self._episode_metadata)

        fps = float(episode.fps or self.config.fps)
        task = self._resolve_task(episode)
        preferred_task_index = episode.metadata.get("task_index")
        if not isinstance(preferred_task_index, int):
            preferred_task_index = None
        task_index = self._get_or_create_task_index(task, preferred_task_index)

        try:
            frame_list = list(episode.frames())
        except Exception as e:
            raise ConversionError("source", "lerobot-v2", f"Failed to read frames: {e}")

        if not frame_list:
            raise ConversionError("source", "lerobot-v2", "Episode has no frames")

        chunk_index = self._episode_chunk(episode_index)
        episode_id = f"episode_{episode_index:06d}"

        rows: list[dict[str, Any]] = []
        video_frames: dict[str, list[LazyImage]] = {}
        stats: dict[str, ArrayStatsAccumulator | ImageStatsAccumulator] = {
            "timestamp": ArrayStatsAccumulator(),
            "frame_index": ArrayStatsAccumulator(),
            "episode_index": ArrayStatsAccumulator(),
            "index": ArrayStatsAccumulator(),
            "task_index": ArrayStatsAccumulator(),
        }

        for frame_offset, frame in enumerate(frame_list):
            if progress_callback:
                progress_callback(frame_offset, len(frame_list))

            frame_index = int(frame.index)
            timestamp = (
                float(frame.timestamp)
                if frame.timestamp is not None
                else float(frame_offset / fps)
            )
            global_index = self._total_frames + frame_offset

            row: dict[str, Any] = {
                "episode_index": episode_index,
                "frame_index": frame_index,
                "index": global_index,
                "task_index": task_index,
                "timestamp": timestamp,
            }

            for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
                stats[key].update(row[key])

            if frame.state is not None:
                state = np.asarray(frame.state)
                row["observation.state"] = state.tolist()
                stats.setdefault("observation.state", ArrayStatsAccumulator()).update(state)
                self._ensure_array_feature("observation.state", state)

            if frame.action is not None:
                action = np.asarray(frame.action)
                row["action"] = action.tolist()
                stats.setdefault("action", ArrayStatsAccumulator()).update(action)
                self._ensure_array_feature("action", action)

            for cam_name, lazy_image in frame.images.items():
                video_key = self._map_camera_name(cam_name)
                video_frames.setdefault(video_key, []).append(lazy_image)
                self._ensure_video_feature(video_key, lazy_image, fps)
                if cam_name not in self._cameras:
                    self._cameras[cam_name] = CameraInfo(
                        name=cam_name,
                        height=lazy_image.height,
                        width=lazy_image.width,
                        channels=lazy_image.channels,
                    )

                image_array = lazy_image.load()
                stats.setdefault(video_key, ImageStatsAccumulator()).update(image_array)
                lazy_image.clear_cache()

            rows.append(row)

        data_dir = output_path / "data" / f"chunk-{chunk_index:03d}"
        data_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = data_dir / f"{episode_id}.parquet"

        try:
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, parquet_path)
        except Exception as e:
            raise ConversionError("source", "lerobot-v2", f"Failed to write parquet: {e}")

        for video_key, frames in video_frames.items():
            if not frames:
                continue

            video_dir = output_path / "videos" / f"chunk-{chunk_index:03d}" / video_key
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"{episode_id}.mp4"
            first_frame = frames[0]

            try:
                self._video_encoder.encode_frames(
                    iter(frames),
                    video_path,
                    fps=fps,
                    width=first_frame.width,
                    height=first_frame.height,
                )
            except Exception as e:
                raise ConversionError("source", "lerobot-v2", f"Failed to encode video: {e}")

        self._episode_metadata.append(
            {
                "episode_index": episode_index,
                "task_index": task_index,
                "tasks": [task],
                "length": len(frame_list),
            }
        )
        self._episode_stats.append(
            {
                "episode_index": episode_index,
                "stats": {
                    name: accumulator.to_serializable()
                    for name, accumulator in stats.items()
                    if accumulator.to_serializable()
                },
            }
        )
        self._total_frames += len(frame_list)
        self._total_videos += len(video_frames)

    def write_dataset(
        self,
        episodes: Iterator[Episode],
        output_path: Path,
        dataset_info: DatasetInfo | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> None:
        """Write a full dataset in legacy LeRobot v2 layout."""
        self._reset_state()
        output_path = Path(output_path)

        if dataset_info is not None:
            if dataset_info.inferred_fps:
                self.config.fps = dataset_info.inferred_fps
            if dataset_info.inferred_robot_type:
                self.config.robot_type = dataset_info.inferred_robot_type

        for episode_idx, episode in enumerate(episodes):
            if progress_callback:
                progress_callback(episode_idx, episode.episode_id)
            self.write_episode(episode, output_path, episode_index=episode_idx)

        final_info = dataset_info or DatasetInfo(
            path=output_path,
            format="lerobot-v2",
            num_episodes=len(self._episode_metadata),
            total_frames=self._total_frames,
            inferred_fps=self.config.fps,
            inferred_robot_type=self.config.robot_type,
            cameras=self._cameras,
        )
        self.finalize(output_path, final_info)

    def finalize(self, output_path: Path, dataset_info: DatasetInfo) -> None:
        """Write legacy LeRobot metadata files."""
        output_path = Path(output_path)
        meta_dir = output_path / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        total_episodes = (
            len(self._episode_metadata) if self._episode_metadata else dataset_info.num_episodes
        )
        total_frames = self._total_frames if self._total_frames > 0 else dataset_info.total_frames
        total_chunks = math.ceil(total_episodes / self.config.chunks_size) if total_episodes else 0
        fps = float(self.config.fps or dataset_info.inferred_fps or 30.0)

        info: dict[str, Any] = {
            "codebase_version": LEGACY_CODEBASE_VERSION,
            "robot_type": self.config.robot_type or dataset_info.inferred_robot_type or "unknown",
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": len(self._task_metadata),
            "total_videos": self._total_videos,
            "total_chunks": total_chunks,
            "chunks_size": self.config.chunks_size,
            "fps": fps,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": LEGACY_DATA_PATH,
            "video_path": LEGACY_VIDEO_PATH,
            "features": self._features,
        }
        if self.config.repo_id:
            info["repo_id"] = self.config.repo_id

        with open(meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)

        with open(meta_dir / "tasks.jsonl", "w") as f:
            for task in sorted(self._task_metadata, key=lambda item: int(item["task_index"])):
                f.write(json.dumps(task) + "\n")

        with open(meta_dir / "episodes.jsonl", "w") as f:
            for episode_meta in sorted(
                self._episode_metadata, key=lambda item: int(item["episode_index"])
            ):
                f.write(json.dumps(episode_meta) + "\n")

        with open(meta_dir / "episodes_stats.jsonl", "w") as f:
            for episode_stats in sorted(
                self._episode_stats, key=lambda item: int(item["episode_index"])
            ):
                f.write(json.dumps(episode_stats) + "\n")
