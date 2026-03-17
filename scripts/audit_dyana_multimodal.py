# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from PIL import Image, ImageOps

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.dyana import (
    DyanaMotionTokenTransform,
    DyanaMotionViewTransform,
    DyanaTargetCropTransform,
)
from gr00t.experiment.data_config import load_data_config


@dataclass
class Args:
    dataset_path: str = "./dyana_data"
    data_config: str = "dyana_motion_crop_token_11f_18d"
    embodiment_tag: str = "dyana_hand_task"
    video_backend: str = "decord"
    sample_count: int = 128
    seed: int = 7
    save_examples_dir: str = ""
    save_examples_limit: int = 8


def clone_step_data(step_data: dict) -> dict:
    cloned = {}
    for key, value in step_data.items():
        if isinstance(value, np.ndarray):
            cloned[key] = value.copy()
        elif isinstance(value, list):
            cloned[key] = list(value)
        else:
            cloned[key] = value
    return cloned


def save_example(example_path: Path, processed: dict):
    panels = []
    for key in ("video.ego_view", "video.motion_view", "video.target_crop"):
        if key not in processed:
            continue
        frame = processed[key][-1]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        panels.append(Image.fromarray(frame))
    if not panels:
        return

    strip = Image.new("RGB", (sum(img.width for img in panels), max(img.height for img in panels)))
    offset = 0
    for image in panels:
        strip.paste(ImageOps.contain(image, (image.width, image.height)), (offset, 0))
        offset += image.width
    strip.save(example_path)


def main(args: Args):
    data_config = load_data_config(args.data_config)
    modality_config = data_config.modality_config()
    transforms = data_config.transform()

    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality_config,
        transforms=None,
        embodiment_tag=args.embodiment_tag,
        video_backend=args.video_backend,
    )
    transforms.set_metadata(dataset.metadata)
    transforms.eval()

    pre_concat_transforms = []
    motion_transform = None
    crop_transform = None
    token_transform = None
    for transform in transforms.transforms:
        if isinstance(transform, ConcatTransform):
            break
        pre_concat_transforms.append(transform)
        if isinstance(transform, DyanaMotionViewTransform):
            motion_transform = transform
        elif isinstance(transform, DyanaTargetCropTransform):
            crop_transform = transform
        elif isinstance(transform, DyanaMotionTokenTransform):
            token_transform = transform

    rng = np.random.default_rng(args.seed)
    total_steps = len(dataset)
    sample_count = min(args.sample_count, total_steps)
    sample_indices = rng.choice(total_steps, size=sample_count, replace=False)

    empty_motion_sequences = 0
    motion_sequences = 0
    fallback_sequences = 0
    crop_sequences = 0
    token_counts: dict[str, int] = {}

    save_dir = Path(args.save_examples_dir) if args.save_examples_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    saved_examples = 0
    for sample_idx in sample_indices:
        processed = clone_step_data(dataset[int(sample_idx)])
        for transform in pre_concat_transforms:
            processed = transform(processed)

        if motion_transform is not None:
            empty_motion_sequences += motion_transform.last_empty_sequences
            motion_sequences += motion_transform.last_sequence_count
        if crop_transform is not None:
            fallback_sequences += crop_transform.last_fallback_sequences
            crop_sequences += crop_transform.last_sequence_count
        if token_transform is not None:
            token_key = data_config.language_keys[0]
            token_value = processed.get(token_key, [])
            if isinstance(token_value, list) and token_value:
                token = token_value[0]
            else:
                token = str(token_value)
            token_counts[token] = token_counts.get(token, 0) + 1

        if save_dir is not None and saved_examples < args.save_examples_limit:
            save_example(save_dir / f"sample_{saved_examples:03d}.png", processed)
            saved_examples += 1

    print(f"data_config={args.data_config}")
    print(f"sample_count={sample_count}")
    if motion_sequences > 0:
        print(
            f"motion_map_empty_rate={empty_motion_sequences / motion_sequences:.4f} "
            f"({empty_motion_sequences}/{motion_sequences})"
        )
    else:
        print("motion_map_empty_rate=NA")
    if crop_sequences > 0:
        print(
            f"target_crop_fallback_rate={fallback_sequences / crop_sequences:.4f} "
            f"({fallback_sequences}/{crop_sequences})"
        )
    else:
        print("target_crop_fallback_rate=NA")
    if token_counts:
        print("motion_token_counts=")
        for token, count in sorted(token_counts.items()):
            print(f"  {token}: {count}")
    if save_dir is not None:
        print(f"saved_examples_dir={save_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
