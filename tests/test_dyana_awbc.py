from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.rl.mixed_dataset import DyanaMixedChunkDataset
from gr00t.rl.protocol import StepFeedback
from gr00t.rl.reward import compute_episode_rewards, repeat_factor_from_zscore
from gr00t.rl.rollout_dataset import DyanaRolloutChunkDataset


def make_feedback(
    decision_step: int,
    done: bool,
    success: bool,
    min_distance_to_target: float,
    min_joint_to_surface_distance: float,
) -> StepFeedback:
    return StepFeedback(
        episode_id=1,
        repeat=0,
        decision_step=decision_step,
        done=done,
        success=success,
        task_type="linear",
        current_frame_index=decision_step * 30,
        success_index=decision_step * 10,
        min_distance_to_target=min_distance_to_target,
        min_joint_to_surface_distance=min_joint_to_surface_distance,
    )


def make_metadata() -> DatasetMetadata:
    return DatasetMetadata.model_validate(
        {
            "statistics": {
                "state": {
                    "left_hand": {
                        "max": [1.0] * 18,
                        "min": [-1.0] * 18,
                        "mean": [0.0] * 18,
                        "std": [1.0] * 18,
                        "q01": [-0.9] * 18,
                        "q99": [0.9] * 18,
                    }
                },
                "action": {
                    "left_hand": {
                        "max": [1.0] * 18,
                        "min": [-1.0] * 18,
                        "mean": [0.0] * 18,
                        "std": [1.0] * 18,
                        "q01": [-0.9] * 18,
                        "q99": [0.9] * 18,
                    }
                },
            },
            "modalities": {
                "video": {
                    "ego_view": {
                        "resolution": [64, 64],
                        "channels": 3,
                        "fps": 30,
                    }
                },
                "state": {
                    "left_hand": {
                        "absolute": True,
                        "rotation_type": None,
                        "shape": [18],
                        "continuous": True,
                    }
                },
                "action": {
                    "left_hand": {
                        "absolute": True,
                        "rotation_type": None,
                        "shape": [18],
                        "continuous": True,
                    }
                },
            },
            "embodiment_tag": EmbodimentTag.DYANA_HAND_TASK.value,
        }
    )


class DummyDataset(Dataset):
    def __init__(self, prefix: str, length: int):
        self.prefix = prefix
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        return f"{self.prefix}-{index % self.length}"

    def set_epoch(self, epoch: int):
        self.epoch = epoch


def test_step_feedback_round_trip_preserves_fields():
    original = make_feedback(
        decision_step=2,
        done=True,
        success=False,
        min_distance_to_target=0.12,
        min_joint_to_surface_distance=0.05,
    )

    payload = original.to_dict()
    restored = StepFeedback.from_payload(payload)

    assert restored == original


def test_compute_episode_rewards_tracks_progress_and_terminal_bonus():
    feedbacks = [
        make_feedback(0, False, False, 0.50, 0.30),
        make_feedback(1, False, False, 0.35, 0.18),
        make_feedback(2, True, True, 0.20, 0.10),
    ]

    rewards, returns = compute_episode_rewards(feedbacks)

    assert rewards.shape == (3,)
    assert returns.shape == (3,)
    assert rewards[0] < 0.0
    assert rewards[1] > rewards[0]
    assert rewards[2] > 0.5
    assert np.isclose(returns[0], rewards.sum())


def test_repeat_factor_from_zscore_matches_awbc_schedule():
    assert repeat_factor_from_zscore(1.1) == 4
    assert repeat_factor_from_zscore(0.2) == 2
    assert repeat_factor_from_zscore(-0.4) == 1
    assert repeat_factor_from_zscore(-1.2) == 0


def test_dyana_mixed_chunk_dataset_enforces_three_to_one_pattern():
    demo_dataset = DummyDataset(prefix="demo", length=12)
    rollout_dataset = DummyDataset(prefix="rollout", length=4)
    mixed_dataset = DyanaMixedChunkDataset(demo_dataset, rollout_dataset)

    first_eight = [mixed_dataset[index] for index in range(8)]

    assert first_eight == [
        "demo-0",
        "demo-1",
        "demo-2",
        "rollout-0",
        "demo-3",
        "demo-4",
        "demo-5",
        "rollout-1",
    ]


def test_rollout_chunk_dataset_reads_npz_and_repeats_chunk_refs(tmp_path: Path):
    episode_dir = tmp_path / "episode_000000"
    episode_dir.mkdir()

    with open(episode_dir / "episode_metrics.json", "w", encoding="utf-8") as handle:
        handle.write(
            '{"task_type": "linear", "episode_return": 0.4, "success": true, "score": 0.8}'
        )

    np.savez_compressed(
        episode_dir / "chunks.npz",
        **{
            "video.ego_view": np.zeros((1, 11, 64, 64, 3), dtype=np.uint8),
            "state.left_hand": np.zeros((1, 11, 18), dtype=np.float32),
            "action.left_hand": np.zeros((1, 10, 18), dtype=np.float32),
            "annotation.human.action.task_description": np.asarray(["motion=linear"]),
            "decision_step": np.asarray([0], dtype=np.int32),
            "reward": np.asarray([0.1], dtype=np.float32),
            "return": np.asarray([0.1], dtype=np.float32),
            "done": np.asarray([True], dtype=np.bool_),
        },
    )

    dataset = DyanaRolloutChunkDataset(
        rollout_dir=tmp_path,
        transforms=None,
        dataset_metadata=make_metadata(),
    )

    assert len(dataset) == 2
    sample = dataset[0]
    assert sample["video.ego_view"].shape == (11, 64, 64, 3)
    assert sample["state.left_hand"].shape == (11, 18)
    assert sample["action.left_hand"].shape == (10, 18)
    assert sample["annotation.human.action.task_description"] == "motion=linear"
