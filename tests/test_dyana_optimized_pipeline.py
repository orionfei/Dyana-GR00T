import json
from pathlib import Path

import pytest

from gr00t.data.dyana_subset import (
    FilteredStepDataset,
    build_dyana_sampling_artifacts,
    canonicalize_dyana_task_type,
)
from gr00t.utils.experiment import assert_adapter_checkpoint_artifacts


TASK_PROMPTS = {
    "linear": "Grab the object in the video that is making a straight motion",
    "circular": "Grab the object in the video that is making a circular motion",
    "harmonic": "Grab the object in the video that is doing simple harmonic motion",
}


class DummyTransform:
    def __init__(self):
        self.training = True
        self.metadata = None

    def set_metadata(self, metadata):
        self.metadata = metadata

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __call__(self, data):
        return {
            "value": int(data["value"]),
            "training": self.training,
            "tag": self.metadata["tag"],
        }


class DummyBaseDataset:
    def __init__(self):
        self.metadata = {"tag": "dummy"}
        self.tag = "dummy"
        self.dataset_path = Path("dummy_dataset")
        self.dataset_name = "dummy_dataset"
        self.epoch = 0

    def get_step_data(self, trajectory_id, base_index):
        return {"value": trajectory_id * 100 + base_index}

    def set_epoch(self, epoch):
        self.epoch = epoch


def _write_dyana_episodes(meta_dir: Path) -> dict[int, int]:
    episode_specs = [
        (0, "linear", 6),
        (1, "linear", 8),
        (2, "linear", 10),
        (3, "circular", 6),
        (4, "circular", 8),
        (5, "circular", 10),
        (6, "harmonic", 6),
        (7, "harmonic", 8),
        (8, "harmonic", 10),
    ]
    lengths = {}
    with open(meta_dir / "episodes.jsonl", "w", encoding="utf-8") as handle:
        for episode_id, task_type, length in episode_specs:
            handle.write(
                json.dumps(
                    {
                        "episode_index": episode_id,
                        "tasks": [TASK_PROMPTS[task_type], "valid"],
                        "length": length,
                    }
                )
                + "\n"
            )
            lengths[episode_id] = length
    return lengths


def test_canonicalize_dyana_task_type_handles_prompt_variants():
    assert canonicalize_dyana_task_type("linear") == "linear"
    assert canonicalize_dyana_task_type(TASK_PROMPTS["linear"]) == "linear"
    assert canonicalize_dyana_task_type(TASK_PROMPTS["circular"]) == "circular"
    assert canonicalize_dyana_task_type(TASK_PROMPTS["harmonic"]) == "harmonic"
    with pytest.raises(ValueError):
        canonicalize_dyana_task_type("random walk")


def test_build_dyana_sampling_artifacts_is_stratified_and_disjoint(tmp_path: Path):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir(parents=True)
    lengths = _write_dyana_episodes(meta_dir)

    artifacts = build_dyana_sampling_artifacts(
        dataset_path=tmp_path,
        trajectory_lengths=lengths,
        train_episodes_per_task=2,
        holdout_episodes_per_task=1,
        base_index_stride=2,
        task_repeat_factors={"linear": 2, "circular": 1, "harmonic": 1},
        subset_seed=7,
        stage="pilot",
    )

    assert artifacts.summary["task_source"] == "meta/episodes.jsonl"
    assert artifacts.summary["train"]["episodes_per_task"] == {
        "linear": 2,
        "circular": 2,
        "harmonic": 2,
    }
    assert artifacts.summary["holdout"]["episodes_per_task"] == {
        "linear": 1,
        "circular": 1,
        "harmonic": 1,
    }

    for task_type in ("linear", "circular", "harmonic"):
        assert not (
            set(artifacts.train_episode_ids_by_task[task_type])
            & set(artifacts.holdout_episode_ids_by_task[task_type])
        )

    assert all(base_index % 2 == 0 for _, base_index in artifacts.train_manifest)
    assert all(base_index % 2 == 0 for _, base_index in artifacts.holdout_manifest)

    train_summary = artifacts.summary["train"]
    assert (
        train_summary["step_counts_after_repeat"]["linear"]
        == train_summary["step_counts_after_stride"]["linear"] * 2
    )
    assert (
        train_summary["step_counts_after_repeat"]["circular"]
        == train_summary["step_counts_after_stride"]["circular"]
    )
    assert (
        train_summary["step_counts_after_repeat"]["harmonic"]
        == train_summary["step_counts_after_stride"]["harmonic"]
    )


def test_filtered_step_dataset_uses_manifest_and_own_transform():
    base_dataset = DummyBaseDataset()
    dataset = FilteredStepDataset(
        base_dataset=base_dataset,
        manifest=[(2, 0), (2, 4), (5, 6)],
        transforms=DummyTransform(),
    )

    dataset.train()
    first = dataset[1]
    assert first == {"value": 204, "training": True, "tag": "dummy"}

    dataset.eval()
    second = dataset[2]
    assert second == {"value": 506, "training": False, "tag": "dummy"}

    dataset.set_epoch(3)
    assert base_dataset.epoch == 3
    assert len(dataset) == 3
    assert dataset.all_steps == [(2, 0), (2, 4), (5, 6)]


def test_assert_adapter_checkpoint_artifacts_accepts_expected_files(tmp_path: Path):
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
    assert_adapter_checkpoint_artifacts(tmp_path)

    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    (broken_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        assert_adapter_checkpoint_artifacts(broken_dir)
