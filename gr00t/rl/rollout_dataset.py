# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from gr00t.data.schema import DatasetMetadata

from .reward import EpisodeReturnRecord, score_episode_records


@dataclass(frozen=True)
class RolloutChunkRef:
    episode_dir: Path
    chunk_index: int
    task_type: str
    episode_return: float
    z_score: float
    repeat_factor: int


class DyanaRolloutChunkDataset(Dataset):
    def __init__(
        self,
        rollout_dir: str | Path,
        transforms,
        dataset_metadata: DatasetMetadata,
        video_key: str = "video.ego_view",
        state_key: str = "state.left_hand",
        action_key: str = "action.left_hand",
        language_key: str = "annotation.human.action.task_description",
    ):
        self.rollout_dir = Path(rollout_dir)
        self.transforms = transforms
        self.video_key = video_key
        self.state_key = state_key
        self.action_key = action_key
        self.language_key = language_key
        self.metadata = copy.deepcopy(dataset_metadata)
        self.tag = (
            self.metadata.embodiment_tag.value
            if hasattr(self.metadata.embodiment_tag, "value")
            else str(self.metadata.embodiment_tag)
        )
        self._epoch = 0
        self._episode_cache: dict[Path, dict[str, np.ndarray]] = {}
        self._manifest = self._build_manifest()
        self._sync_video_resolution_from_rollouts()
        if self.transforms is not None:
            self.transforms.set_metadata(self.metadata)
            self.transforms.train()

    def _build_manifest(self) -> list[RolloutChunkRef]:
        episode_dirs = sorted(path for path in self.rollout_dir.glob("episode_*") if path.is_dir())
        episode_records = []
        for episode_dir in episode_dirs:
            metrics_path = episode_dir / "episode_metrics.json"
            chunks_path = episode_dir / "chunks.npz"
            if not metrics_path.exists() or not chunks_path.exists():
                continue
            with open(metrics_path, "r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            episode_records.append(
                EpisodeReturnRecord(
                    episode_dir=episode_dir,
                    task_type=str(metrics["task_type"]),
                    episode_return=float(metrics["episode_return"]),
                    success=bool(metrics["success"]),
                    score=float(metrics["score"]),
                )
            )

        manifest: list[RolloutChunkRef] = []
        for record in score_episode_records(episode_records):
            if record.repeat_factor <= 0:
                continue
            with np.load(record.episode_dir / "chunks.npz", allow_pickle=False) as arrays:
                chunk_count = int(arrays["decision_step"].shape[0])
            for chunk_index in range(chunk_count):
                for _ in range(record.repeat_factor):
                    manifest.append(
                        RolloutChunkRef(
                            episode_dir=record.episode_dir,
                            chunk_index=chunk_index,
                            task_type=record.task_type,
                            episode_return=record.episode_return,
                            z_score=record.z_score,
                            repeat_factor=record.repeat_factor,
                        )
                    )
        return manifest

    def _sync_video_resolution_from_rollouts(self) -> None:
        if not self._manifest:
            return
        first_episode = self._manifest[0].episode_dir
        arrays = self._load_episode_arrays(first_episode)
        video = arrays[self.video_key]
        if video.ndim != 5:
            raise ValueError(f"Expected rollout video to be [N,T,H,W,C], got {video.shape}")
        height = int(video.shape[-3])
        width = int(video.shape[-2])
        sub_key = self.video_key.split(".", 1)[1]
        if sub_key in self.metadata.modalities.video:
            self.metadata.modalities.video[sub_key].resolution = (width, height)

    def _load_episode_arrays(self, episode_dir: Path) -> dict[str, np.ndarray]:
        cached = self._episode_cache.get(episode_dir)
        if cached is not None:
            return cached
        with np.load(episode_dir / "chunks.npz", allow_pickle=False) as arrays:
            cached = {key: arrays[key].copy() for key in arrays.files}
        self._episode_cache[episode_dir] = cached
        return cached

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not self._manifest:
            raise IndexError("Rollout dataset is empty")
        ref = self._manifest[(index + self._epoch) % len(self._manifest)]
        arrays = self._load_episode_arrays(ref.episode_dir)
        sample = {
            self.video_key: arrays[self.video_key][ref.chunk_index].copy(),
            self.state_key: arrays[self.state_key][ref.chunk_index].copy(),
            self.action_key: arrays[self.action_key][ref.chunk_index].copy(),
            self.language_key: arrays[self.language_key][ref.chunk_index].tolist(),
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample
