# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from pydantic import Field, PrivateAttr

from gr00t.data.transform.base import ModalityTransform


def _ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


class DyanaMotionTokenTransform(ModalityTransform):
    """Compress Dyana language supervision to a stable motion token."""

    apply_to: list[str] = Field(..., description="Language keys to canonicalize.")
    unknown_token: str = Field(default="motion=unknown")

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    data[key] = self._canonicalize_text(str(value.item()))
                else:
                    data[key] = [
                        self._canonicalize_text(str(item)) for item in value.reshape(-1).tolist()
                    ]
            elif isinstance(value, (list, tuple)):
                data[key] = [self._canonicalize_text(str(item)) for item in value]
            else:
                data[key] = self._canonicalize_text(str(value))
        return data

    def _canonicalize_text(self, text: str) -> str:
        lowered = text.lower()
        if "harmonic" in lowered:
            return "motion=harmonic"
        if "circular" in lowered:
            return "motion=circular"
        if "straight" in lowered or "linear" in lowered:
            return "motion=linear"
        return self.unknown_token


class DyanaMotionViewTransform(ModalityTransform):
    """Create a motion-focused view from the ego video using frame differencing."""

    apply_to: list[str] = Field(default_factory=list, description="Unused; kept for compatibility.")
    source_key: str = Field(default="video.ego_view")
    target_key: str = Field(default="video.motion_view")
    gaussian_blur_kernel: int = Field(default=5)
    diff_threshold: int = Field(default=12)
    temporal_decay: float = Field(default=0.75)

    _last_empty_sequences: int = PrivateAttr(default=0)
    _last_sequence_count: int = PrivateAttr(default=0)

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.source_key not in data:
            return data
        video = data[self.source_key]
        self._last_empty_sequences = 0
        self._last_sequence_count = 0
        data[self.target_key] = self._build_motion_view(video)
        return data

    @property
    def last_empty_sequences(self) -> int:
        return self._last_empty_sequences

    @property
    def last_sequence_count(self) -> int:
        return self._last_sequence_count

    def _build_motion_view(self, video: np.ndarray) -> np.ndarray:
        if video.ndim == 4:
            motion_view, is_empty = self._build_single_motion_view(video)
            self._last_empty_sequences = int(is_empty)
            self._last_sequence_count = 1
            return motion_view
        if video.ndim == 5:
            clips = []
            empty_count = 0
            for clip in video:
                motion_view, is_empty = self._build_single_motion_view(clip)
                clips.append(motion_view)
                empty_count += int(is_empty)
            self._last_empty_sequences = empty_count
            self._last_sequence_count = len(clips)
            return np.stack(clips, axis=0)
        raise ValueError(f"Expected video to have 4 or 5 dimensions, got {video.ndim}")

    def _build_single_motion_view(self, clip: np.ndarray) -> tuple[np.ndarray, bool]:
        frames = np.asarray(clip, dtype=np.uint8)
        num_frames, height, width = frames.shape[:3]
        motion_frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
        trail = np.zeros((height, width), dtype=np.float32)
        blur_kernel = _ensure_odd(max(self.gaussian_blur_kernel, 1))

        for idx in range(1, num_frames):
            diff = cv2.absdiff(gray_frames[idx], gray_frames[idx - 1])
            if blur_kernel > 1:
                diff = cv2.GaussianBlur(diff, (blur_kernel, blur_kernel), 0)
            diff = np.where(diff >= self.diff_threshold, diff, 0).astype(np.float32)
            trail = np.maximum(diff, trail * self.temporal_decay)
            motion_uint8 = np.clip(trail, 0, 255).astype(np.uint8)
            motion_frames[idx] = np.repeat(motion_uint8[..., None], 3, axis=-1)

        is_empty = bool(motion_frames.max() == 0)
        return motion_frames, is_empty


class DyanaTargetCropTransform(ModalityTransform):
    """Create a target-centric crop using motion localization inside the clip."""

    apply_to: list[str] = Field(default_factory=list, description="Unused; kept for compatibility.")
    source_key: str = Field(default="video.ego_view")
    target_key: str = Field(default="video.target_crop")
    motion_key: str | None = Field(default="video.motion_view")
    min_motion_pixels: int = Field(default=25)
    crop_size: int = Field(default=96)
    expand_ratio: float = Field(default=1.6)

    _last_fallback_sequences: int = PrivateAttr(default=0)
    _last_sequence_count: int = PrivateAttr(default=0)

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.source_key not in data:
            return data
        source_video = data[self.source_key]
        motion_video = data.get(self.motion_key) if self.motion_key is not None else None
        self._last_fallback_sequences = 0
        self._last_sequence_count = 0
        data[self.target_key] = self._build_target_crop(source_video, motion_video)
        return data

    @property
    def last_fallback_sequences(self) -> int:
        return self._last_fallback_sequences

    @property
    def last_sequence_count(self) -> int:
        return self._last_sequence_count

    def _build_target_crop(
        self, source_video: np.ndarray, motion_video: np.ndarray | None
    ) -> np.ndarray:
        if source_video.ndim == 4:
            cropped, used_fallback = self._build_single_target_crop(source_video, motion_video)
            self._last_fallback_sequences = int(used_fallback)
            self._last_sequence_count = 1
            return cropped
        if source_video.ndim == 5:
            crops = []
            fallback_count = 0
            for idx, clip in enumerate(source_video):
                clip_motion = None if motion_video is None else motion_video[idx]
                cropped, used_fallback = self._build_single_target_crop(clip, clip_motion)
                crops.append(cropped)
                fallback_count += int(used_fallback)
            self._last_fallback_sequences = fallback_count
            self._last_sequence_count = len(crops)
            return np.stack(crops, axis=0)
        raise ValueError(f"Expected video to have 4 or 5 dimensions, got {source_video.ndim}")

    def _build_single_target_crop(
        self, clip: np.ndarray, motion_clip: np.ndarray | None
    ) -> tuple[np.ndarray, bool]:
        clip = np.asarray(clip, dtype=np.uint8)
        height, width = clip.shape[1:3]
        center_x = width // 2
        center_y = height // 2
        side = min(max(self.crop_size, 1), height, width)
        used_fallback = True

        motion_map = self._get_motion_map(clip, motion_clip)
        active_pixels = np.argwhere(motion_map > 0)
        if active_pixels.shape[0] >= self.min_motion_pixels:
            y_min, x_min = active_pixels.min(axis=0)
            y_max, x_max = active_pixels.max(axis=0)
            center_y = int(round((y_min + y_max) / 2))
            center_x = int(round((x_min + x_max) / 2))
            box_height = max(int(y_max - y_min + 1), self.crop_size)
            box_width = max(int(x_max - x_min + 1), self.crop_size)
            side = int(round(max(box_height, box_width) * self.expand_ratio))
            side = min(max(side, self.crop_size), height, width)
            used_fallback = False

        x0, y0, x1, y1 = self._square_box(center_x, center_y, side, width, height)
        crops = []
        for frame in clip:
            crop = frame[y0:y1, x0:x1]
            resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
            crops.append(resized)
        return np.stack(crops, axis=0), used_fallback

    def _get_motion_map(self, clip: np.ndarray, motion_clip: np.ndarray | None) -> np.ndarray:
        if motion_clip is not None:
            motion_intensity = motion_clip.astype(np.float32).mean(axis=-1)
            aggregated = motion_intensity.max(axis=0)
        else:
            grayscale = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in clip]
            diffs = []
            for idx in range(1, len(grayscale)):
                diff = cv2.absdiff(grayscale[idx], grayscale[idx - 1])
                diffs.append(diff.astype(np.float32))
            aggregated = np.max(np.stack(diffs, axis=0), axis=0) if diffs else np.zeros(
                clip.shape[1:3], dtype=np.float32
            )

        if aggregated.max() <= 0:
            return np.zeros_like(aggregated, dtype=np.uint8)
        threshold = max(8.0, float(aggregated.mean() + aggregated.std()))
        return (aggregated >= threshold).astype(np.uint8)

    @staticmethod
    def _square_box(
        center_x: int, center_y: int, side: int, width: int, height: int
    ) -> tuple[int, int, int, int]:
        half = side // 2
        x0 = max(center_x - half, 0)
        y0 = max(center_y - half, 0)
        x1 = min(x0 + side, width)
        y1 = min(y0 + side, height)
        x0 = max(x1 - side, 0)
        y0 = max(y1 - side, 0)
        return x0, y0, x1, y1
