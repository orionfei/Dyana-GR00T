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

# python scripts/gr00t_finetune.py   --dataset-path /data1/yfl_data/Dyana-GR00T/dyana_data   
# --output-dir /data1/yfl_data/Dyana-GR00T/checkpoints/dyana_hand_task_lora --embodiment-tag dyana_hand_task  
# --video-backend decord   --num-gpus 4   --batch-size 4  --max-steps 100   --save-steps 50  
# --lora-rank 32  --lora-alpha 64   --lora-dropout 0.05  --report-to tensorboard 

import os
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import torch
import tyro
from transformers import TrainingArguments

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.dyana import (
    DyanaMotionTokenTransform,
    DyanaMotionViewTransform,
    DyanaTargetCropTransform,
)
from gr00t.experiment.data_config import load_data_config
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING, GR00TTransform
from gr00t.utils.peft import get_lora_model


DYANA_PHASE1_CONFIG_SPECS = {
    "dyana_lora_11f_18d": {"video_views": 1, "motion_view": False, "target_crop": False, "token": False},
    "dyana_motion_token_11f_18d": {"video_views": 1, "motion_view": False, "target_crop": False, "token": True},
    "dyana_motion_view_11f_18d": {"video_views": 2, "motion_view": True, "target_crop": False, "token": False},
    "dyana_motion_view_token_11f_18d": {"video_views": 2, "motion_view": True, "target_crop": False, "token": True},
    "dyana_target_crop_11f_18d": {"video_views": 2, "motion_view": False, "target_crop": True, "token": False},
    "dyana_target_crop_token_11f_18d": {"video_views": 2, "motion_view": False, "target_crop": True, "token": True},
    "dyana_motion_crop_11f_18d": {"video_views": 3, "motion_view": True, "target_crop": True, "token": False},
    "dyana_motion_crop_token_11f_18d": {"video_views": 3, "motion_view": True, "target_crop": True, "token": True},
}


def _sync_video_resolution_with_actual_data(dataset: LeRobotSingleDataset) -> None:
    """Align dataset video metadata resolution with actual decoded frames.

    Some custom datasets have stale `meta/info.json` video shapes (e.g. 256x256)
    while raw videos are stored at higher resolutions. Video transforms validate
    against metadata resolution, so mismatch causes training to fail early.
    """
    if len(dataset.trajectory_ids) == 0:
        return

    traj_id = int(dataset.trajectory_ids[0])
    raw_step = dataset.get_step_data(traj_id, base_index=0)
    updated = False

    for video_key in dataset.modality_keys.get("video", []):
        if video_key not in raw_step:
            continue
        frames = raw_step[video_key]  # [T, H, W, C]
        height, width = int(frames.shape[-3]), int(frames.shape[-2])
        actual_resolution = (width, height)

        sub_key = video_key.split(".", 1)[1]
        if sub_key not in dataset.metadata.modalities.video:
            continue
        expected_resolution = dataset.metadata.modalities.video[sub_key].resolution

        if expected_resolution != actual_resolution:
            print(
                f"Updating video resolution metadata for {video_key}: "
                f"{expected_resolution} -> {actual_resolution}"
            )
            dataset.metadata.modalities.video[sub_key].resolution = actual_resolution
            updated = True

    if updated:
        dataset.set_transforms_metadata(dataset.metadata)


def _strict_preflight_dataset(
    dataset: LeRobotSingleDataset,
    modality_configs: dict,
    transforms,
    data_config_name: str,
    embodiment_tag: EmbodimentTag,
) -> None:
    """Fail-fast validation for task-critical dataset/model contract."""
    assert len(dataset.trajectory_ids) > 0, "Dataset has no trajectories"

    expected_tag = embodiment_tag.value if hasattr(embodiment_tag, "value") else str(embodiment_tag)
    actual_tag = (
        dataset.metadata.embodiment_tag.value
        if hasattr(dataset.metadata.embodiment_tag, "value")
        else str(dataset.metadata.embodiment_tag)
    )
    assert (
        actual_tag == expected_tag
    ), f"Embodiment tag mismatch: metadata={actual_tag}, expected={expected_tag}"

    obs_horizon = len(modality_configs["state"].delta_indices)
    action_horizon = len(modality_configs["action"].delta_indices)

    traj_id = int(dataset.trajectory_ids[0])
    raw_step = dataset.get_step_data(traj_id, base_index=0)

    # Ensure expected keys exist and have valid raw shapes.
    for video_key in modality_configs["video"].modality_keys:
        assert video_key in raw_step, f"Missing video key in sample: {video_key}"
        frames = raw_step[video_key]
        assert frames.ndim == 4, f"{video_key} must be [T,H,W,C], got {frames.shape}"
        assert frames.shape[0] == obs_horizon, (
            f"{video_key} horizon mismatch: got {frames.shape[0]}, expected {obs_horizon}"
        )
        assert frames.shape[-1] == 3, f"{video_key} must have 3 channels, got shape {frames.shape}"
        sub_key = video_key.split(".", 1)[1]
        assert (
            sub_key in dataset.metadata.modalities.video
        ), f"Missing video metadata for key '{sub_key}'"
        expected_resolution = tuple(dataset.metadata.modalities.video[sub_key].resolution)
        actual_resolution = (int(frames.shape[-2]), int(frames.shape[-3]))
        assert expected_resolution == actual_resolution, (
            f"Video resolution mismatch for {video_key}: metadata={expected_resolution}, "
            f"actual={actual_resolution}. Please sync metadata before training."
        )

    state_total_dim = 0
    for state_key in modality_configs["state"].modality_keys:
        assert state_key in raw_step, f"Missing state key in sample: {state_key}"
        state = raw_step[state_key]
        assert state.ndim == 2, f"{state_key} must be [T,D], got {state.shape}"
        assert state.shape[0] == obs_horizon, (
            f"{state_key} horizon mismatch: got {state.shape[0]}, expected {obs_horizon}"
        )
        sub_key = state_key.split(".", 1)[1]
        assert (
            sub_key in dataset.metadata.modalities.state
        ), f"Missing state metadata for key '{sub_key}'"
        expected_dim = int(dataset.metadata.modalities.state[sub_key].shape[0])
        assert state.shape[1] == expected_dim, (
            f"{state_key} dim mismatch: data={state.shape[1]}, metadata={expected_dim}"
        )
        state_total_dim += expected_dim

    action_total_dim = 0
    for action_key in modality_configs["action"].modality_keys:
        assert action_key in raw_step, f"Missing action key in sample: {action_key}"
        action = raw_step[action_key]
        assert action.ndim == 2, f"{action_key} must be [T,D], got {action.shape}"
        assert action.shape[0] == action_horizon, (
            f"{action_key} horizon mismatch: got {action.shape[0]}, expected {action_horizon}"
        )
        sub_key = action_key.split(".", 1)[1]
        assert (
            sub_key in dataset.metadata.modalities.action
        ), f"Missing action metadata for key '{sub_key}'"
        expected_dim = int(dataset.metadata.modalities.action[sub_key].shape[0])
        assert action.shape[1] == expected_dim, (
            f"{action_key} dim mismatch: data={action.shape[1]}, metadata={expected_dim}"
        )
        action_total_dim += expected_dim

    assert (
        hasattr(transforms, "transforms") and len(transforms.transforms) > 0
    ), "No transforms found"
    last_transform = transforms.transforms[-1]
    assert isinstance(last_transform, GR00TTransform), "Last transform must be GR00TTransform"
    assert (
        last_transform.state_horizon == obs_horizon
    ), f"GR00TTransform state_horizon={last_transform.state_horizon}, expected={obs_horizon}"
    assert (
        last_transform.action_horizon == action_horizon
    ), f"GR00TTransform action_horizon={last_transform.action_horizon}, expected={action_horizon}"
    assert (
        last_transform.max_action_dim >= action_total_dim
    ), f"max_action_dim={last_transform.max_action_dim} < required action dim {action_total_dim}"

    # Task-specific hard constraints for this project.
    if data_config_name in DYANA_PHASE1_CONFIG_SPECS:
        dyana_spec = DYANA_PHASE1_CONFIG_SPECS[data_config_name]
        concat_transform = next(
            (transform for transform in transforms.transforms if isinstance(transform, ConcatTransform)),
            None,
        )
        assert concat_transform is not None, "ConcatTransform is required for Dyana configs"
        assert (
            modality_configs["video"].modality_keys == ["video.ego_view"]
        ), f"Dyana configs must load only raw ego video, got {modality_configs['video'].modality_keys}"
        assert obs_horizon == 11, f"Expected 11-frame observation horizon, got {obs_horizon}"
        assert action_horizon == 10, f"Expected 10-step action horizon, got {action_horizon}"
        assert state_total_dim == 18, f"Expected 18D state, got {state_total_dim}"
        assert action_total_dim == 18, f"Expected 18D action, got {action_total_dim}"
        assert (
            last_transform.max_action_dim == 18
        ), f"Expected max_action_dim=18 for {data_config_name}, got {last_transform.max_action_dim}"
        assert len(concat_transform.video_concat_order) == dyana_spec["video_views"], (
            f"Unexpected Dyana video stream count for {data_config_name}: "
            f"{len(concat_transform.video_concat_order)} vs {dyana_spec['video_views']}"
        )

        has_motion_view = any(
            isinstance(transform, DyanaMotionViewTransform) for transform in transforms.transforms
        )
        has_target_crop = any(
            isinstance(transform, DyanaTargetCropTransform) for transform in transforms.transforms
        )
        has_token = any(
            isinstance(transform, DyanaMotionTokenTransform) for transform in transforms.transforms
        )
        assert has_motion_view == dyana_spec["motion_view"], (
            f"Dyana motion-view transform mismatch for {data_config_name}: "
            f"expected {dyana_spec['motion_view']}, got {has_motion_view}"
        )
        assert has_target_crop == dyana_spec["target_crop"], (
            f"Dyana target-crop transform mismatch for {data_config_name}: "
            f"expected {dyana_spec['target_crop']}, got {has_target_crop}"
        )
        assert has_token == dyana_spec["token"], (
            f"Dyana motion-token transform mismatch for {data_config_name}: "
            f"expected {dyana_spec['token']}, got {has_token}"
        )

    print(
        f"Preflight OK | traj={traj_id} obs_horizon={obs_horizon} action_horizon={action_horizon} "
        f"state_dim={state_total_dim} action_dim={action_total_dim}"
    )


@dataclass
class ArgsConfig:
    """Configuration for GR00T model fine-tuning."""

    # Dataset parameters
    dataset_path: List[str]
    """Path to the dataset directory or directories, we assume all datasets have the same data config"""

    output_dir: str = "/data1/yfl_data/Dyana-GR00T/checkpoints/dyana_hand_task_lora"
    """Directory to save model checkpoints."""

    data_config: str = "dyana_lora_11f_18d"
    """
    Data configuration to use for training.
    Options:
    - Built-in configs: Use predefined config names like 'so100', 'fourier_gr1_arms_only', 'unitree_g1'.
    - External configs: Use 'module:ClassName' format to load custom configs from external files. e.g. 'my_dir.my_configs:RobotConfig'
    See gr00t/experiment/data_config.py for more details.
    """

    # Training parameters
    batch_size: int = 4
    """Batch size per GPU for training."""

    max_steps: int = 20000
    """Maximum number of training steps."""

    num_gpus: int = 4
    """Number of GPUs to use for training."""

    save_steps: int = 2000
    """Number of steps between saving checkpoints."""

    # Model parameters
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path or HuggingFace model ID for the base model."""

    tune_llm: bool = False
    """Whether to fine-tune the language model backbone."""

    tune_visual: bool = False
    """Whether to fine-tune the vision tower."""

    tune_projector: bool = True
    """Whether to fine-tune the projector."""

    tune_diffusion_model: bool = True
    """Whether to fine-tune the diffusion model."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    # Advanced training parameters
    learning_rate: float = 1e-4
    """Learning rate for training."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    lora_rank: int = 0
    """Rank for the LORA model. If 0, no LORA will be used."""

    lora_alpha: int = 16
    """Alpha value for the LORA model."""

    lora_dropout: float = 0.1
    """Dropout rate for the LORA model."""

    lora_full_model: bool = False
    """Whether to use the full model for LORA. If False, only the action head will be trained."""

    dataloader_num_workers: int = 12
    """Number of workers for data loading per GPU."""

    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps for training."""

    dataloader_prefetch_factor: int = 4
    """Prefetch factor for data loading."""

    report_to: Literal["wandb", "tensorboard", "azure_ml"] = "wandb"
    """Where to report training metrics (e.g., 'wandb', 'tensorboard', 'azure_ml')."""

    # Data loading parameters
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "dyana_hand_task"
    """Embodiment tag to use for training. e.g. 'dyana_hand_task', 'new_embodiment', 'gr1'"""

    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "torchcodec"
    """Video backend to use for training. [torchcodec, decord, torchvision_av]"""

    # Mixture dataset parameters
    balance_dataset_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, we will balance the dataset weights, by multiplying the total trajectory to each dataset"""

    # Mixture dataset parameters
    balance_trajectory_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, sample trajectories within a dataset weighted by their length; otherwise, equal weighting."""

    strict_preflight: bool = True
    """If True, run fail-fast checks on dataset keys/shapes/horizons/dimensions before training."""


#####################################################################################
# Helper functions
#####################################################################################


def _copy_partial_action_expert_weights(old_dict, new_dict, old_dim, new_dim):
    """
    Copy weights with partial dimension matching for action_dim changes.
    NOTE(Youliang): this is a very experimental implementation to handle action_dim changes. TODO: improve this.
    """
    def _copy_overlap(src: torch.Tensor, dst: torch.Tensor) -> int:
        if src.dim() != dst.dim():
            return 0
        overlap_shape = tuple(min(int(s), int(d)) for s, d in zip(src.shape, dst.shape))
        if len(overlap_shape) == 0:
            return 0
        slices = tuple(slice(0, n) for n in overlap_shape)
        dst[slices] = src[slices]
        copied = 1
        for n in overlap_shape:
            copied *= n
        return copied

    total_params = copied_params = random_params = 0

    for key, new_tensor in new_dict.items():
        total_params += new_tensor.numel()
        old_tensor = old_dict.get(key)
        if old_tensor is None:
            random_params += new_tensor.numel()
            continue

        if old_tensor.shape == new_tensor.shape:
            new_tensor.copy_(old_tensor)
            copied_params += new_tensor.numel()
            continue

        if "action_encoder" in key or "action_decoder" in key:
            copied = _copy_overlap(old_tensor, new_tensor)
            copied_params += copied
            random_params += new_tensor.numel() - copied
        else:
            random_params += new_tensor.numel()

    assert total_params == copied_params + random_params, "Parameter count mismatch"
    random_percentage = (random_params / total_params) * 100 if total_params > 0 else 0
    print(
        f"Weight copy stats: {copied_params:,} copied, {random_params:,} random ({random_percentage:.1f}% randomly initialized)"
    )
    if new_dim > old_dim:
        print(f"Action dimensions {old_dim + 1}-{new_dim} will be learned from scratch")
    elif new_dim < old_dim:
        print(f"Action dimension reduced: kept first {new_dim} dims from previous {old_dim}-dim head")
    else:
        print("Action dimension unchanged")
    return new_dict


#####################################################################################
# main training function
#####################################################################################


def main(config: ArgsConfig):
    """Main training function."""
    # ------------ step 1: load dataset ------------
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    # 1.1 modality configs and transforms
    data_config_cls = load_data_config(config.data_config)
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    # 1.2 data loader: we will use either single dataset or mixture dataset
    if len(config.dataset_path) == 1:
        train_dataset = LeRobotSingleDataset(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,  # This overrides the dataset embodiment tag for finetuning.
            video_backend=config.video_backend,
        )
        _sync_video_resolution_with_actual_data(train_dataset)
        if config.strict_preflight:
            _strict_preflight_dataset(
                train_dataset,
                modality_configs,
                transforms,
                data_config_name=config.data_config,
                embodiment_tag=embodiment_tag,
            )
    else:
        single_datasets = []
        for p in config.dataset_path:
            assert os.path.exists(p), f"Dataset path {p} does not exist"
            ## We use the same transforms, modality configs, and embodiment tag for all datasets here,
            ## in reality, you can use dataset from different modalities and embodiment tags
            dataset = LeRobotSingleDataset(
                dataset_path=p,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=config.video_backend,
            )
            _sync_video_resolution_with_actual_data(dataset)
            if config.strict_preflight:
                _strict_preflight_dataset(
                    dataset,
                    modality_configs,
                    transforms,
                    data_config_name=config.data_config,
                    embodiment_tag=embodiment_tag,
                )
            single_datasets.append(dataset)

        train_dataset = LeRobotMixtureDataset(
            data_mixture=[
                (dataset, 1.0)  # we will use equal weights for all datasets
                for dataset in single_datasets
            ],
            mode="train",
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=42,
            metadata_config={
                "percentile_mixing_method": "weighted_average",
            },
        )
        print(f"Loaded {len(single_datasets)} datasets, with {config.dataset_path} ")

    global_batch_size = config.num_gpus * config.batch_size * config.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(train_dataset) / global_batch_size)
    planned_epochs = config.max_steps / steps_per_epoch
    print(
        f"Training scale | dataset_steps={len(train_dataset)} global_batch={global_batch_size} "
        f"steps_per_epoch={steps_per_epoch} planned_epochs={planned_epochs:.4f}"
    )
    if planned_epochs < 0.1:
        print(
            "Warning: planned training is less than 0.1 epoch. "
            "This is usually too short to judge LoRA quality."
        )

    # ------------ step 2: load model ------------
    # First, get the data config to determine action horizon
    data_action_horizon = len(data_config_cls.action_indices)

    # Assert that the last transform is a GR00TTransform and has max_action_dim
    assert (
        hasattr(transforms, "transforms") and len(transforms.transforms) > 0
    ), "No transforms found"
    last_transform = transforms.transforms[-1]

    assert isinstance(last_transform, GR00TTransform), "Last transform must be GR00TTransform"
    assert hasattr(last_transform, "max_action_dim"), "GR00TTransform must have max_action_dim"
    data_max_action_dim = last_transform.max_action_dim

    # Load model
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=config.tune_llm,  # backbone's LLM
        tune_visual=config.tune_visual,  # backbone's vision tower
        tune_projector=config.tune_projector,  # action head's projector
        tune_diffusion_model=config.tune_diffusion_model,  # action head's DiT
    )

    # Update action_horizon and max_action_dim to match data config
    # Need to recreate action head with correct config since it was initialized with old config
    action_horizon_mismatch = data_action_horizon != model.action_head.config.action_horizon
    action_dim_mismatch = data_max_action_dim != model.action_head.config.action_dim

    if action_horizon_mismatch or action_dim_mismatch:
        # Store old values for logging
        old_action_horizon = model.action_head.config.action_horizon
        old_action_dim = model.action_head.config.action_dim
        print(
            f"Recreating action head with action_horizon {data_action_horizon} (was {old_action_horizon})"
        )
        if action_dim_mismatch:
            print(f"Updating max_action_dim {data_max_action_dim} (was {old_action_dim})")

        # Update the action head config (need to copy to avoid modifying original)
        import copy

        new_action_head_config = copy.deepcopy(model.action_head.config)
        new_action_head_config.action_horizon = data_action_horizon
        new_action_head_config.action_dim = data_max_action_dim

        # Import the FlowmatchingActionHead class
        from gr00t.model.action_head.flow_matching_action_head import (
            FlowmatchingActionHead,
        )

        # Create new action head with updated config
        new_action_head = FlowmatchingActionHead(new_action_head_config)

        # Copy the weights from the old action head to the new one
        if not action_dim_mismatch:
            print("Copying weights from old action head (compatible dimensions)")
            new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)
        else:
            if data_max_action_dim > old_action_dim:
                print(
                    f"Partial weight copy: copying first {old_action_dim} dimensions, "
                    f"initializing last {data_max_action_dim - old_action_dim} dimensions randomly"
                )
            else:
                print(
                    f"Partial weight copy: reducing action dim from {old_action_dim} "
                    f"to {data_max_action_dim} and copying overlapping weights"
                )
            new_action_head.state_dict().update(
                _copy_partial_action_expert_weights(
                    model.action_head.state_dict(),
                    new_action_head.state_dict(),
                    old_action_dim,
                    data_max_action_dim,
                )
            )

        # Replace the action head
        model.action_head = new_action_head

        # Update model config AND the action_head_cfg dictionary that gets saved
        model.config.action_horizon = data_action_horizon
        model.action_horizon = data_action_horizon
        model.config.action_head_cfg["action_horizon"] = data_action_horizon
        model.config.action_head_cfg["action_dim"] = data_max_action_dim

        # Update the main model's action_dim for validation (critical for validate_inputs)
        model.config.action_dim = data_max_action_dim
        model.action_dim = data_max_action_dim

        # Set trainable parameters for the new action head
        model.action_head.set_trainable_parameters(
            tune_projector=config.tune_projector, tune_diffusion_model=config.tune_diffusion_model
        )

    # Set the model's compute_dtype to bfloat16
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=not config.lora_full_model,
            tune_projector=config.tune_projector,
        )
    else:
        print("Warning: lora_rank=0, this run will not train LoRA adapters.")

    # 2.1 modify training args
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=config.dataloader_prefetch_factor,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        # evaluation_strategy="no",
        save_total_limit=5,
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    # 2.2 run experiment
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )

    # 2.3 run experiment
    experiment.train()


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)

    # Print the tyro config
    print("\n" + "=" * 50)
    print("GR00T FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Validate GPU configuration
    assert (
        config.num_gpus <= available_gpus
    ), f"Number of GPUs requested ({config.num_gpus}) is greater than the available GPUs ({available_gpus})"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print(f"Respecting CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    if config.num_gpus == 1:
        # Single-GPU mode: only set a default device mask when user did not provide one.
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Run the script normally
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            # Multi-GPU mode - use torchrun
            script_path = Path(__file__).absolute()

            # Use subprocess.run instead of os.system
            raw_args_list = sys.argv[1:]
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",  # default to 1 node for now
                str(script_path),
                *raw_args_list,
            ]

            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            # Detach torchrun into its own session so background jobs launched via
            # `nohup ... &` do not receive SIGHUP from the original shell session.
            sys.exit(subprocess.run(cmd, env=env, start_new_session=True).returncode)
