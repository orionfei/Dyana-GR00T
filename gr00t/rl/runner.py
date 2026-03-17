# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Iterable, Optional

import torch
from transformers import TrainingArguments, set_seed

from gr00t.experiment.trainer import DualBrainTrainer
from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import DefaultDataCollator, GR00TTransform
from gr00t.utils.experiment import CheckpointFormatCallback, safe_save_model_for_hf_trainer
from gr00t.utils.peft import get_lora_model


def _copy_partial_action_expert_weights(old_dict, new_dict, old_dim, new_dim):
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
        f"Weight copy stats: {copied_params:,} copied, {random_params:,} random "
        f"({random_percentage:.1f}% randomly initialized)"
    )
    if new_dim > old_dim:
        print(f"Action dimensions {old_dim + 1}-{new_dim} will be learned from scratch")
    elif new_dim < old_dim:
        print(f"Action dimension reduced: kept first {new_dim} dims from previous {old_dim}-dim head")
    else:
        print("Action dimension unchanged")
    return new_dict


def _align_action_head_to_data_config(
    model,
    data_action_horizon: int,
    data_max_action_dim: int,
    tune_projector: bool,
    tune_diffusion_model: bool,
):
    action_horizon_mismatch = data_action_horizon != model.action_head.config.action_horizon
    action_dim_mismatch = data_max_action_dim != model.action_head.config.action_dim
    if not action_horizon_mismatch and not action_dim_mismatch:
        return model

    old_action_horizon = model.action_head.config.action_horizon
    old_action_dim = model.action_head.config.action_dim
    print(
        f"Recreating action head with action_horizon {data_action_horizon} "
        f"(was {old_action_horizon})"
    )
    if action_dim_mismatch:
        print(f"Updating max_action_dim {data_max_action_dim} (was {old_action_dim})")

    new_action_head_config = copy.deepcopy(model.action_head.config)
    new_action_head_config.action_horizon = data_action_horizon
    new_action_head_config.action_dim = data_max_action_dim
    new_action_head = FlowmatchingActionHead(new_action_head_config)

    if not action_dim_mismatch:
        new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)
    else:
        new_action_head.state_dict().update(
            _copy_partial_action_expert_weights(
                model.action_head.state_dict(),
                new_action_head.state_dict(),
                old_action_dim,
                data_max_action_dim,
            )
        )

    model.action_head = new_action_head
    model.config.action_horizon = data_action_horizon
    model.action_horizon = data_action_horizon
    model.config.action_head_cfg["action_horizon"] = data_action_horizon
    model.config.action_head_cfg["action_dim"] = data_max_action_dim
    model.config.action_dim = data_max_action_dim
    model.action_dim = data_max_action_dim
    model.action_head.set_trainable_parameters(
        tune_projector=tune_projector,
        tune_diffusion_model=tune_diffusion_model,
    )
    return model


def _set_adapter_trainable(model, modules_to_save: Iterable[str] | None):
    modules_to_save = list(modules_to_save or [])
    for name, parameter in model.named_parameters():
        should_train = "lora_" in name or any(module_name in name for module_name in modules_to_save)
        parameter.requires_grad = should_train


def load_awbc_trainable_model(
    init_checkpoint: str,
    data_transform: GR00TTransform,
    tune_llm: bool = False,
    tune_visual: bool = False,
    tune_projector: bool = True,
    tune_diffusion_model: bool = True,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    lora_full_model: bool = False,
):
    assert hasattr(data_transform, "action_horizon"), "Expected data_transform to be a GR00TTransform"
    data_action_horizon = int(data_transform.action_horizon)
    data_max_action_dim = int(data_transform.max_action_dim)

    init_path = Path(init_checkpoint)
    adapter_config_path = init_path / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path, "r", encoding="utf-8") as handle:
            adapter_config = json.load(handle)
        base_model_path = adapter_config["base_model_name_or_path"]
        print(f"Loading AWBC init checkpoint from adapter: {init_checkpoint}")
        print(f"Loading base model for adapter checkpoint from: {base_model_path}")
        model = GR00T_N1_5.from_pretrained(
            base_model_path,
            tune_llm=tune_llm,
            tune_visual=tune_visual,
            tune_projector=tune_projector,
            tune_diffusion_model=tune_diffusion_model,
        )
        model = _align_action_head_to_data_config(
            model=model,
            data_action_horizon=data_action_horizon,
            data_max_action_dim=data_max_action_dim,
            tune_projector=tune_projector,
            tune_diffusion_model=tune_diffusion_model,
        )
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("peft is required to continue training LoRA checkpoints") from exc

        try:
            model = PeftModel.from_pretrained(model, str(init_path), is_trainable=True)
        except TypeError:
            model = PeftModel.from_pretrained(model, str(init_path))
            _set_adapter_trainable(model, adapter_config.get("modules_to_save"))
        else:
            _set_adapter_trainable(model, adapter_config.get("modules_to_save"))
    else:
        print(f"Loading AWBC init checkpoint as base/full model: {init_checkpoint}")
        model = GR00T_N1_5.from_pretrained(
            init_checkpoint,
            tune_llm=tune_llm,
            tune_visual=tune_visual,
            tune_projector=tune_projector,
            tune_diffusion_model=tune_diffusion_model,
        )
        model = _align_action_head_to_data_config(
            model=model,
            data_action_horizon=data_action_horizon,
            data_max_action_dim=data_max_action_dim,
            tune_projector=tune_projector,
            tune_diffusion_model=tune_diffusion_model,
        )
        if lora_rank > 0:
            model = get_lora_model(
                model,
                rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                action_head_only=not lora_full_model,
                tune_projector=tune_projector,
            )

    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"
    model.train()
    return model


class AWBCTrainRunner:
    def __init__(
        self,
        model,
        training_args: TrainingArguments,
        train_dataset,
        dataset_metadata,
        resume_from_checkpoint: bool = False,
    ):
        self.training_args = training_args
        self.output_dir = Path(training_args.output_dir)
        self.exp_cfg_dir = self.output_dir / "experiment_cfg"
        self.exp_cfg_dir.mkdir(parents=True, exist_ok=True)
        self.resume_from_checkpoint = resume_from_checkpoint
        self.train_dataset = train_dataset

        training_args.run_name = (
            training_args.output_dir.split("/")[-1]
            if training_args.run_name is None
            else training_args.run_name
        )
        print(f"Run name: {training_args.run_name}")

        data_collator = DefaultDataCollator()
        compute_dtype = torch.float16 if training_args.bf16 else torch.float32
        set_seed(training_args.seed)

        self.trainer = self.create_trainer(
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            compute_dtype=compute_dtype,
        )

        self.rank = int(os.environ.get("RANK", 0))
        if self.rank == 0:
            tag = (
                dataset_metadata.embodiment_tag.value
                if hasattr(dataset_metadata.embodiment_tag, "value")
                else str(dataset_metadata.embodiment_tag)
            )
            with open(self.exp_cfg_dir / "metadata.json", "w", encoding="utf-8") as handle:
                json.dump({tag: dataset_metadata.model_dump(mode="json")}, handle, indent=4)

        report_to = training_args.report_to
        if report_to == "wandb":
            if "WANDB_PROJECT" not in os.environ:
                os.environ["WANDB_PROJECT"] = "gr00t-training"
            if "WANDB_RUN_ID" not in os.environ:
                runtime_id = os.environ.get("RUNTIME_ID")
                if runtime_id:
                    os.environ["WANDB_RUN_ID"] = runtime_id
            os.environ["WANDB_DIR"] = training_args.output_dir
            with open(self.output_dir / "wandb_config.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "project": os.environ.get("WANDB_PROJECT", ""),
                        "run_id": os.environ.get("WANDB_RUN_ID", ""),
                    },
                    handle,
                )
            training_args.report_to = ["wandb"]
        elif report_to == "azure_ml":
            print("azure_ml logging is enabled.")
        else:
            tensorboard_dir = Path(training_args.output_dir) / "runs"
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            print(f"TensorBoard logs will be saved to: {tensorboard_dir}")
            training_args.report_to = ["tensorboard"]

    def create_trainer(
        self,
        model,
        training_args,
        train_dataset,
        data_collator,
        compute_dtype,
        global_batch_size=None,
    ):
        if global_batch_size is not None:
            batch_size = training_args.per_device_train_batch_size
            num_gpus = max(torch.cuda.device_count(), 1)
            grad_acc = max(1, global_batch_size // (batch_size * num_gpus))
            training_args.gradient_accumulation_steps = grad_acc
            print(
                f"Set global batch size to {global_batch_size}, "
                f"set gradient accumulation steps to {grad_acc}"
            )

        trainer = DualBrainTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            compute_dtype=compute_dtype,
        )
        trainer.add_callback(
            CheckpointFormatCallback(
                run_name=training_args.run_name,
                exp_cfg_dir=self.exp_cfg_dir,
            )
        )

        train_dl_len = len(trainer.get_train_dataloader())
        print(
            f"train dataloader length: {train_dl_len}\n"
            f"train dataset length: {len(trainer.train_dataset)}\n"
            f"GPU memory before training: "
            f"{torch.cuda.memory_allocated() / 1024 / 1024 / 1024} GB",
            flush=True,
        )
        return trainer

    def train(self):
        self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
        self.trainer.save_state()
        safe_save_model_for_hf_trainer(
            trainer=self.trainer,
            output_dir=self.training_args.output_dir,
        )
