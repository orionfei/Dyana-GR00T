import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from transformers import TrainingArguments

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING, GR00TTransform
from gr00t.rl import (
    AWBCTrainRunner,
    DyanaMixedChunkDataset,
    DyanaRolloutChunkDataset,
    load_awbc_trainable_model,
)


def _sync_video_resolution_with_actual_data(dataset: LeRobotSingleDataset) -> None:
    if len(dataset.trajectory_ids) == 0:
        return

    traj_id = int(dataset.trajectory_ids[0])
    raw_step = dataset.get_step_data(traj_id, base_index=0)
    updated = False

    for video_key in dataset.modality_keys.get("video", []):
        if video_key not in raw_step:
            continue
        frames = raw_step[video_key]
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


@dataclass
class ArgsConfig:
    demo_dataset: str
    rollout_dir: str
    init_checkpoint: str
    output_dir: str

    data_config: str = "dyana_motion_crop_token_11f_18d"
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "dyana_hand_task"
    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "torchcodec"

    batch_size: int = 4
    num_gpus: int = 1
    num_train_epochs: float = 1.0
    learning_rate: float = 5e-5
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 8
    dataloader_prefetch_factor: int = 4
    logging_steps: float = 10.0
    seed: int = 42

    demo_ratio: float = 0.75
    rollout_ratio: float = 0.25

    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_full_model: bool = False

    report_to: Literal["wandb", "tensorboard", "azure_ml"] = "wandb"
    resume: bool = False


def main(config: ArgsConfig):
    embodiment_tag = EmbodimentTag(config.embodiment_tag)
    data_config_cls = load_data_config(config.data_config)
    modality_configs = data_config_cls.modality_config()
    demo_transforms = data_config_cls.transform()
    rollout_transforms = data_config_cls.transform()

    demo_dataset = LeRobotSingleDataset(
        dataset_path=config.demo_dataset,
        modality_configs=modality_configs,
        transforms=demo_transforms,
        embodiment_tag=embodiment_tag,
        video_backend=config.video_backend,
    )
    _sync_video_resolution_with_actual_data(demo_dataset)

    rollout_dataset = DyanaRolloutChunkDataset(
        rollout_dir=config.rollout_dir,
        transforms=rollout_transforms,
        dataset_metadata=demo_dataset.metadata,
    )
    if len(rollout_dataset) == 0 and config.rollout_ratio > 0:
        raise ValueError(f"Rollout dataset is empty: {config.rollout_dir}")

    train_dataset = DyanaMixedChunkDataset(
        demo_dataset=demo_dataset,
        rollout_dataset=rollout_dataset,
        demo_ratio=config.demo_ratio,
        rollout_ratio=config.rollout_ratio,
    )

    steps_per_epoch = math.ceil(
        len(train_dataset) / max(config.batch_size * config.num_gpus * config.gradient_accumulation_steps, 1)
    )
    print(
        f"AWBC training scale | train_steps={len(train_dataset)} demo_steps={len(demo_dataset)} "
        f"rollout_steps={len(rollout_dataset)} steps_per_epoch={steps_per_epoch}"
    )

    assert hasattr(demo_transforms, "transforms") and len(demo_transforms.transforms) > 0
    last_transform = demo_transforms.transforms[-1]
    assert isinstance(last_transform, GR00TTransform), "Last transform must be GR00TTransform"

    model = load_awbc_trainable_model(
        init_checkpoint=config.init_checkpoint,
        data_transform=last_transform,
        tune_llm=config.tune_llm,
        tune_visual=config.tune_visual,
        tune_projector=config.tune_projector,
        tune_diffusion_model=config.tune_diffusion_model,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_full_model=config.lora_full_model,
    )

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
        logging_steps=config.logging_steps,
        num_train_epochs=config.num_train_epochs,
        max_steps=-1,
        save_strategy="epoch",
        save_total_limit=3,
        report_to=config.report_to,
        seed=config.seed,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    experiment = AWBCTrainRunner(
        train_dataset=train_dataset,
        dataset_metadata=demo_dataset.metadata,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )
    experiment.train()


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)

    print("\n" + "=" * 50)
    print("DYANA AWBC FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    assert (
        config.num_gpus <= available_gpus
    ), f"Number of GPUs requested ({config.num_gpus}) is greater than the available GPUs ({available_gpus})"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print(f"Respecting CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    if config.num_gpus == 1:
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            script_path = Path(__file__).absolute()
            raw_args_list = sys.argv[1:]
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",
                str(script_path),
                *raw_args_list,
            ]

            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env, start_new_session=True).returncode)
