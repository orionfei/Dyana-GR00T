# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .protocol import StepFeedback

DISTANCE_CLIP = 0.05
GRASP_DISTANCE_CLIP = 0.02
STEP_PENALTY = 0.02
SUCCESS_BONUS = 1.0
FAILURE_PENALTY = 0.3


@dataclass
class EpisodeReturnRecord:
    episode_dir: Path
    task_type: str
    episode_return: float
    success: bool
    score: float
    z_score: float = 0.0
    repeat_factor: int = 0


def _normalize_delta(delta: float, clip_value: float) -> float:
    return float(np.clip(delta, -clip_value, clip_value) / clip_value)


def compute_chunk_reward(
    previous_feedback: StepFeedback | None,
    current_feedback: StepFeedback,
) -> float:
    progress_dist = 0.0
    progress_grasp = 0.0
    if previous_feedback is not None:
        progress_dist = _normalize_delta(
            previous_feedback.min_distance_to_target - current_feedback.min_distance_to_target,
            DISTANCE_CLIP,
        )
        progress_grasp = _normalize_delta(
            previous_feedback.min_joint_to_surface_distance
            - current_feedback.min_joint_to_surface_distance,
            GRASP_DISTANCE_CLIP,
        )

    reward = 0.4 * progress_dist + 0.3 * progress_grasp - STEP_PENALTY
    if current_feedback.done and current_feedback.success:
        reward += SUCCESS_BONUS
    elif current_feedback.done and not current_feedback.success:
        reward -= FAILURE_PENALTY
    return float(reward)


def compute_returns(rewards: Sequence[float]) -> np.ndarray:
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running += float(rewards[idx])
        returns[idx] = running
    return returns


def compute_episode_rewards(feedbacks: Sequence[StepFeedback]) -> tuple[np.ndarray, np.ndarray]:
    rewards = []
    previous_feedback = None
    for feedback in feedbacks:
        rewards.append(compute_chunk_reward(previous_feedback, feedback))
        previous_feedback = feedback
    rewards_array = np.asarray(rewards, dtype=np.float32)
    returns = compute_returns(rewards_array.tolist())
    return rewards_array, returns


def repeat_factor_from_zscore(z_score: float) -> int:
    if z_score >= 1.0:
        return 4
    if z_score >= 0.0:
        return 2
    if z_score >= -1.0:
        return 1
    return 0


def score_episode_records(records: Iterable[EpisodeReturnRecord]) -> list[EpisodeReturnRecord]:
    scored_records = list(records)
    if not scored_records:
        return scored_records

    grouped: dict[str, list[EpisodeReturnRecord]] = {}
    for record in scored_records:
        grouped.setdefault(record.task_type, []).append(record)

    for task_records in grouped.values():
        returns = np.asarray([record.episode_return for record in task_records], dtype=np.float32)
        mean = float(returns.mean())
        std = float(returns.std())
        if std < 1e-6:
            std = 1.0
        for record in task_records:
            record.z_score = float((record.episode_return - mean) / std)
            record.repeat_factor = repeat_factor_from_zscore(record.z_score)

    return scored_records
