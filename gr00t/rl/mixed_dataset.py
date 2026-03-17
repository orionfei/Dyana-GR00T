# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

from torch.utils.data import Dataset


class DyanaMixedChunkDataset(Dataset):
    def __init__(
        self,
        demo_dataset: Dataset,
        rollout_dataset: Dataset,
        demo_ratio: float = 0.75,
        rollout_ratio: float = 0.25,
    ):
        if len(demo_dataset) == 0:
            raise ValueError("demo_dataset must be non-empty")
        self.demo_dataset = demo_dataset
        self.rollout_dataset = rollout_dataset
        self.demo_ratio = demo_ratio
        self.rollout_ratio = rollout_ratio
        self.metadata = getattr(demo_dataset, "metadata", None)
        self.tag = getattr(demo_dataset, "tag", None)
        self._epoch = 0

        if self.rollout_ratio > 0 and abs((self.demo_ratio + self.rollout_ratio) - 1.0) > 1e-6:
            raise ValueError("demo_ratio and rollout_ratio must sum to 1.0")

        if len(self.rollout_dataset) == 0 or self.rollout_ratio <= 0:
            self._length = len(self.demo_dataset)
        else:
            self._length = max(
                int(math.ceil(len(self.demo_dataset) / self.demo_ratio)),
                int(math.ceil(len(self.rollout_dataset) / self.rollout_ratio)),
            )

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        if hasattr(self.demo_dataset, "set_epoch"):
            self.demo_dataset.set_epoch(epoch)
        if hasattr(self.rollout_dataset, "set_epoch"):
            self.rollout_dataset.set_epoch(epoch)

    def __getitem__(self, index: int):
        if len(self.rollout_dataset) == 0 or self.rollout_ratio <= 0:
            return self.demo_dataset[(index + self._epoch) % len(self.demo_dataset)]

        cycle_index = (index + self._epoch) % self._length
        slot = cycle_index % 4
        if slot == 3:
            rollout_index = ((cycle_index // 4) + self._epoch) % len(self.rollout_dataset)
            return self.rollout_dataset[rollout_index]

        demo_index = (((cycle_index // 4) * 3) + slot + self._epoch) % len(self.demo_dataset)
        return self.demo_dataset[demo_index]
