# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StepFeedback:
    episode_id: int
    repeat: int
    decision_step: int
    done: bool
    success: bool
    task_type: str
    current_frame_index: int
    success_index: int
    min_distance_to_target: float
    min_joint_to_surface_distance: float

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "StepFeedback":
        return cls(
            episode_id=int(payload["episode_id"]),
            repeat=int(payload["repeat"]),
            decision_step=int(payload["decision_step"]),
            done=bool(payload["done"]),
            success=bool(payload["success"]),
            task_type=str(payload["task_type"]),
            current_frame_index=int(payload["current_frame_index"]),
            success_index=int(payload["successIndex"]),
            min_distance_to_target=float(payload["min_distance_to_target"]),
            min_joint_to_surface_distance=float(payload["minJointToSurfaceDistance"]),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["successIndex"] = payload.pop("success_index")
        payload["minJointToSurfaceDistance"] = payload.pop("min_joint_to_surface_distance")
        return payload
