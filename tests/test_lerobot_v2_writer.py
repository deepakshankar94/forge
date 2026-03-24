"""Tests for LeRobot v2 writer."""

import json
from pathlib import Path

import numpy as np
import pytest

from forge.core.models import CameraInfo, Episode, Frame, LazyImage
from forge.formats.lerobot_v2.reader import LeRobotV2Reader
from forge.formats.lerobot_v2.writer import LeRobotV2Writer, LeRobotV2WriterConfig
from forge.formats.registry import FormatRegistry


def _check_dependencies_available() -> bool:
    """Check if all required dependencies are available."""
    try:
        import av  # noqa: F401
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture
def sample_v2_episodes() -> list[Episode]:
    """Create a small set of episodes for legacy writer tests."""

    def make_episode(episode_index: int, length: int) -> Episode:
        task = f"Task {episode_index}"

        def frame_loader():
            for frame_idx in range(length):
                def image_loader(idx: int = frame_idx, ep_idx: int = episode_index):
                    img = np.zeros((32, 32, 3), dtype=np.uint8)
                    img[:, :, 0] = (ep_idx + 1) * 30
                    img[:, :, 1] = idx * 20
                    img[:, :, 2] = np.arange(32, dtype=np.uint8).reshape(1, -1)
                    return img

                yield Frame(
                    index=frame_idx,
                    timestamp=frame_idx / 15.0,
                    images={
                        "camera0": LazyImage(
                            loader=image_loader,
                            height=32,
                            width=32,
                            channels=3,
                        )
                    },
                    state=np.array(
                        [episode_index, frame_idx, frame_idx + 0.5],
                        dtype=np.float32,
                    ),
                    action=np.array(
                        [frame_idx * 0.1, episode_index + 0.25],
                        dtype=np.float32,
                    ),
                )

        return Episode(
            episode_id=f"ep_{episode_index:03d}",
            metadata={"task_index": episode_index},
            language_instruction=task,
            cameras={"camera0": CameraInfo(name="camera0", height=32, width=32)},
            fps=15.0,
            _frame_loader=frame_loader,
        )

    return [make_episode(0, 4), make_episode(1, 3)]


class TestLeRobotV2Writer:
    """Tests for LeRobotV2Writer."""

    def test_writer_registration(self):
        """Test writer registration in the format registry."""
        assert FormatRegistry.has_writer("lerobot-v2") is True

        writer = FormatRegistry.get_writer("lerobot-v2")

        assert isinstance(writer, LeRobotV2Writer)
        assert writer.format_name == "lerobot-v2"

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_write_dataset_and_roundtrip(
        self,
        tmp_path: Path,
        sample_v2_episodes: list[Episode],
    ):
        """Test writing legacy LeRobot v2 layout and reading it back."""
        import pyarrow.parquet as pq

        output_dir = tmp_path / "lerobot_v2"
        writer = LeRobotV2Writer(
            LeRobotV2WriterConfig(fps=15.0, robot_type="test_robot", chunks_size=1000)
        )

        writer.write_dataset(iter(sample_v2_episodes), output_dir)

        assert (output_dir / "meta" / "info.json").exists()
        assert (output_dir / "meta" / "tasks.jsonl").exists()
        assert (output_dir / "meta" / "episodes.jsonl").exists()
        assert (output_dir / "meta" / "episodes_stats.jsonl").exists()

        parquet_path = output_dir / "data" / "chunk-000" / "episode_000000.parquet"
        video_path = (
            output_dir
            / "videos"
            / "chunk-000"
            / "observation.images.camera0"
            / "episode_000000.mp4"
        )

        assert parquet_path.exists()
        assert video_path.exists()

        with open(output_dir / "meta" / "info.json") as f:
            info = json.load(f)

        assert info["codebase_version"] == "v2.1"
        assert info["robot_type"] == "test_robot"
        assert info["total_episodes"] == 2
        assert info["total_frames"] == 7
        assert info["total_tasks"] == 2
        assert info["total_videos"] == 2
        assert info["chunks_size"] == 1000
        assert info["data_path"] == "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        assert (
            info["video_path"]
            == "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        )
        assert "observation.state" in info["features"]
        assert "action" in info["features"]
        assert "observation.images.camera0" in info["features"]

        tasks = _read_jsonl(output_dir / "meta" / "tasks.jsonl")
        assert tasks == [
            {"task_index": 0, "task": "Task 0"},
            {"task_index": 1, "task": "Task 1"},
        ]

        episodes = _read_jsonl(output_dir / "meta" / "episodes.jsonl")
        assert episodes == [
            {"episode_index": 0, "task_index": 0, "tasks": ["Task 0"], "length": 4},
            {"episode_index": 1, "task_index": 1, "tasks": ["Task 1"], "length": 3},
        ]

        episode_stats = _read_jsonl(output_dir / "meta" / "episodes_stats.jsonl")
        assert len(episode_stats) == 2
        assert episode_stats[0]["episode_index"] == 0
        assert "timestamp" in episode_stats[0]["stats"]
        assert "frame_index" in episode_stats[0]["stats"]
        assert "index" in episode_stats[0]["stats"]
        assert "task_index" in episode_stats[0]["stats"]
        assert "observation.state" in episode_stats[0]["stats"]
        assert "action" in episode_stats[0]["stats"]
        assert "observation.images.camera0" in episode_stats[0]["stats"]

        table = pq.read_table(parquet_path)
        assert table.num_rows == 4
        assert set(table.column_names) == {
            "episode_index",
            "frame_index",
            "index",
            "task_index",
            "timestamp",
            "observation.state",
            "action",
        }

        reader = LeRobotV2Reader()
        inspected = reader.inspect(output_dir)
        assert inspected.format == "lerobot-v2"
        assert inspected.num_episodes == 2
        assert inspected.has_language is True
        assert inspected.sample_language == "Task 0"

        roundtrip_episodes = list(reader.read_episodes(output_dir))
        assert len(roundtrip_episodes) == 2
        assert roundtrip_episodes[0].language_instruction == "Task 0"
        assert roundtrip_episodes[0].metadata["task_index"] == 0

        frames = roundtrip_episodes[0].load_frames()
        assert len(frames) == 4
        assert frames[0].timestamp == pytest.approx(0.0)
        assert np.allclose(frames[2].state, np.array([0.0, 2.0, 2.5], dtype=np.float32))
        assert np.allclose(frames[2].action, np.array([0.2, 0.25], dtype=np.float32))
        assert frames[0].get_image("camera0").shape == (32, 32, 3)
