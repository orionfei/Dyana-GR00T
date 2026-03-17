from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from gr00t.data.dataset import LeRobotSingleDataset


TASK_TYPE_ORDER = ("linear", "circular", "harmonic")


@dataclass(frozen=True)
class DyanaSamplingArtifacts:
    train_manifest: list[tuple[int, int]]
    holdout_manifest: list[tuple[int, int]]
    train_episode_ids_by_task: dict[str, list[int]]
    holdout_episode_ids_by_task: dict[str, list[int]]
    summary: dict[str, Any]


class FilteredStepDataset(Dataset):
    """Expose a filtered `(trajectory_id, base_index)` manifest over a raw LeRobot dataset."""

    def __init__(
        self,
        base_dataset: LeRobotSingleDataset,
        manifest: list[tuple[int, int]],
        transforms=None,
    ):
        self.base_dataset = base_dataset
        self._manifest = [(int(trajectory_id), int(base_index)) for trajectory_id, base_index in manifest]
        self.metadata = copy.deepcopy(base_dataset.metadata)
        self.tag = base_dataset.tag
        self.transforms = copy.deepcopy(transforms if transforms is not None else base_dataset.transforms)
        if self.transforms is not None:
            self.transforms.set_metadata(self.metadata)

    @property
    def all_steps(self) -> list[tuple[int, int]]:
        return self._manifest

    @property
    def dataset_path(self) -> Path:
        return self.base_dataset.dataset_path

    @property
    def dataset_name(self) -> str:
        return self.base_dataset.dataset_name

    def set_transforms_metadata(self, metadata) -> None:
        self.metadata = metadata
        if self.transforms is not None:
            self.transforms.set_metadata(metadata)

    def train(self) -> None:
        if self.transforms is not None:
            self.transforms.train()

    def eval(self) -> None:
        if self.transforms is not None:
            self.transforms.eval()

    def set_epoch(self, epoch: int) -> None:
        self.base_dataset.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self._manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        trajectory_id, base_index = self._manifest[index]
        sample = self.base_dataset.get_step_data(trajectory_id, base_index)
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_dataset, name)


def canonicalize_dyana_task_type(raw_value: Any) -> str:
    text = str(raw_value).strip().lower()
    if not text:
        raise ValueError("Empty task type")
    if text in TASK_TYPE_ORDER:
        return text
    if "straight" in text or "linear" in text:
        return "linear"
    if "circular" in text or "circle" in text:
        return "circular"
    if "harmonic" in text:
        return "harmonic"
    raise ValueError(f"Unsupported Dyana task type: {raw_value}")


def load_dyana_task_index(dataset_path: str | Path) -> tuple[dict[int, str], str]:
    dataset_path = Path(dataset_path)
    task_index: dict[int, str] = {}
    source = "unknown"

    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    if episodes_path.exists():
        with open(episodes_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                episode = json.loads(line)
                trajectory_id = int(episode["episode_index"])
                raw_task = None
                if "task_type" in episode:
                    raw_task = episode["task_type"]
                elif "task" in episode:
                    raw_task = episode["task"]
                elif "tasks" in episode and episode["tasks"]:
                    raw_task = episode["tasks"][0]
                if raw_task is None:
                    continue
                task_index[trajectory_id] = canonicalize_dyana_task_type(raw_task)
        if task_index:
            source = "meta/episodes.jsonl"
            return task_index, source

    unity_meta_dir = dataset_path / "unity_meta"
    if unity_meta_dir.exists():
        for meta_path in sorted(unity_meta_dir.glob("episode_*.json")):
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)

            raw_episode_id = meta.get("episode_id")
            if raw_episode_id is None:
                raw_episode_id = meta_path.stem.split("_")[-1]
            trajectory_id = int(raw_episode_id)

            raw_task = (
                meta.get("task_type")
                or meta.get("task")
                or meta.get("task_description")
                or meta.get("motion_type")
            )
            if raw_task is None:
                continue
            task_index[trajectory_id] = canonicalize_dyana_task_type(raw_task)
        if task_index:
            source = "unity_meta" if source == "unknown" else f"{source}+unity_meta"

    if not task_index:
        raise FileNotFoundError(
            f"Unable to derive Dyana task labels from {dataset_path}. "
            "Expected meta/episodes.jsonl and/or unity_meta/*.json."
        )

    return task_index, source


def build_dyana_sampling_artifacts(
    dataset_path: str | Path,
    trajectory_lengths: dict[int, int],
    train_episodes_per_task: int,
    holdout_episodes_per_task: int,
    base_index_stride: int,
    task_repeat_factors: dict[str, int],
    subset_seed: int,
    stage: str,
) -> DyanaSamplingArtifacts:
    if train_episodes_per_task <= 0:
        raise ValueError("train_episodes_per_task must be > 0")
    if holdout_episodes_per_task <= 0:
        raise ValueError("holdout_episodes_per_task must be > 0")
    if base_index_stride <= 0:
        raise ValueError("base_index_stride must be > 0")

    task_index, task_source = load_dyana_task_index(dataset_path)
    task_to_episode_ids: dict[str, list[int]] = {task_type: [] for task_type in TASK_TYPE_ORDER}
    unlabeled_trajectory_ids: list[int] = []

    for trajectory_id in sorted(int(value) for value in trajectory_lengths):
        task_type = task_index.get(int(trajectory_id))
        if task_type is None:
            unlabeled_trajectory_ids.append(int(trajectory_id))
            continue
        task_to_episode_ids[task_type].append(int(trajectory_id))

    train_episode_ids_by_task: dict[str, list[int]] = {}
    holdout_episode_ids_by_task: dict[str, list[int]] = {}

    for task_offset, task_type in enumerate(TASK_TYPE_ORDER):
        available_ids = list(task_to_episode_ids[task_type])
        rng = random.Random(subset_seed + task_offset)
        rng.shuffle(available_ids)

        required_count = train_episodes_per_task + holdout_episodes_per_task
        if len(available_ids) < required_count:
            raise ValueError(
                f"Not enough Dyana episodes for task '{task_type}': "
                f"required {required_count}, found {len(available_ids)}"
            )

        train_episode_ids_by_task[task_type] = sorted(available_ids[:train_episodes_per_task])
        holdout_episode_ids_by_task[task_type] = sorted(
            available_ids[train_episodes_per_task:required_count]
        )

    train_manifest, train_summary = _build_step_manifest(
        trajectory_lengths=trajectory_lengths,
        episode_ids_by_task=train_episode_ids_by_task,
        base_index_stride=base_index_stride,
        task_repeat_factors=task_repeat_factors,
    )
    holdout_manifest, holdout_summary = _build_step_manifest(
        trajectory_lengths=trajectory_lengths,
        episode_ids_by_task=holdout_episode_ids_by_task,
        base_index_stride=base_index_stride,
        task_repeat_factors={task_type: 1 for task_type in TASK_TYPE_ORDER},
    )

    overlap = {
        task_type: sorted(
            set(train_episode_ids_by_task[task_type]).intersection(holdout_episode_ids_by_task[task_type])
        )
        for task_type in TASK_TYPE_ORDER
    }
    if any(overlap[task_type] for task_type in TASK_TYPE_ORDER):
        raise ValueError(f"Dyana train/holdout episode overlap detected: {overlap}")

    summary = {
        "stage": stage,
        "subset_seed": subset_seed,
        "task_source": task_source,
        "base_index_stride": base_index_stride,
        "task_repeat_factors": {task_type: int(task_repeat_factors[task_type]) for task_type in TASK_TYPE_ORDER},
        "unlabeled_trajectory_count": len(unlabeled_trajectory_ids),
        "train": {
            "episodes_per_task": {
                task_type: len(train_episode_ids_by_task[task_type]) for task_type in TASK_TYPE_ORDER
            },
            "step_counts_before_stride": train_summary["step_counts_before_stride"],
            "step_counts_after_stride": train_summary["step_counts_after_stride"],
            "step_counts_after_repeat": train_summary["step_counts_after_repeat"],
            "total_step_count": len(train_manifest),
        },
        "holdout": {
            "episodes_per_task": {
                task_type: len(holdout_episode_ids_by_task[task_type]) for task_type in TASK_TYPE_ORDER
            },
            "step_counts_before_stride": holdout_summary["step_counts_before_stride"],
            "step_counts_after_stride": holdout_summary["step_counts_after_stride"],
            "step_counts_after_repeat": holdout_summary["step_counts_after_repeat"],
            "total_step_count": len(holdout_manifest),
        },
    }

    return DyanaSamplingArtifacts(
        train_manifest=train_manifest,
        holdout_manifest=holdout_manifest,
        train_episode_ids_by_task=train_episode_ids_by_task,
        holdout_episode_ids_by_task=holdout_episode_ids_by_task,
        summary=summary,
    )


def persist_dyana_sampling_artifacts(
    output_dir: str | Path,
    artifacts: DyanaSamplingArtifacts,
    base_index_stride: int,
    task_repeat_factors: dict[str, int],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_manifest_payload = {
        "base_index_stride": int(base_index_stride),
        "task_repeat_factors": {task_type: int(task_repeat_factors[task_type]) for task_type in TASK_TYPE_ORDER},
        "episodes_by_task": artifacts.train_episode_ids_by_task,
        "step_count": len(artifacts.train_manifest),
        "steps": artifacts.train_manifest,
    }
    holdout_manifest_payload = {
        "base_index_stride": int(base_index_stride),
        "task_repeat_factors": {task_type: 1 for task_type in TASK_TYPE_ORDER},
        "episodes_by_task": artifacts.holdout_episode_ids_by_task,
        "step_count": len(artifacts.holdout_manifest),
        "steps": artifacts.holdout_manifest,
    }

    with open(output_dir / "dyana_train_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(train_manifest_payload, handle, separators=(",", ":"))
    with open(output_dir / "dyana_holdout_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(holdout_manifest_payload, handle, separators=(",", ":"))
    with open(output_dir / "dyana_sampling_summary.json", "w", encoding="utf-8") as handle:
        json.dump(artifacts.summary, handle, indent=2)


def _build_step_manifest(
    trajectory_lengths: dict[int, int],
    episode_ids_by_task: dict[str, list[int]],
    base_index_stride: int,
    task_repeat_factors: dict[str, int],
) -> tuple[list[tuple[int, int]], dict[str, dict[str, int]]]:
    manifest: list[tuple[int, int]] = []
    step_counts_before_stride = {task_type: 0 for task_type in TASK_TYPE_ORDER}
    step_counts_after_stride = {task_type: 0 for task_type in TASK_TYPE_ORDER}
    step_counts_after_repeat = {task_type: 0 for task_type in TASK_TYPE_ORDER}

    for task_type in TASK_TYPE_ORDER:
        repeat_factor = int(task_repeat_factors.get(task_type, 1))
        if repeat_factor <= 0:
            raise ValueError(f"Repeat factor for task '{task_type}' must be > 0")

        for trajectory_id in episode_ids_by_task[task_type]:
            trajectory_length = int(trajectory_lengths[int(trajectory_id)])
            step_counts_before_stride[task_type] += trajectory_length
            filtered_steps = [
                (int(trajectory_id), int(base_index))
                for base_index in range(0, trajectory_length, base_index_stride)
            ]
            step_counts_after_stride[task_type] += len(filtered_steps)
            for _ in range(repeat_factor):
                manifest.extend(filtered_steps)
                step_counts_after_repeat[task_type] += len(filtered_steps)

    return manifest, {
        "step_counts_before_stride": step_counts_before_stride,
        "step_counts_after_stride": step_counts_after_stride,
        "step_counts_after_repeat": step_counts_after_repeat,
    }
