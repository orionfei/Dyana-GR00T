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

import shutil
from pathlib import Path

import torch
from transformers import Trainer, TrainerCallback


def assert_adapter_checkpoint_artifacts(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    adapter_config = output_dir / "adapter_config.json"
    adapter_weight_candidates = [
        output_dir / "adapter_model.bin",
        output_dir / "adapter_model.safetensors",
    ]
    if not adapter_config.exists():
        raise FileNotFoundError(f"Missing LoRA adapter config in checkpoint: {adapter_config}")
    if not any(path.exists() for path in adapter_weight_candidates):
        raise FileNotFoundError(
            "Missing LoRA adapter weights in checkpoint. Expected one of: "
            + ", ".join(str(path) for path in adapter_weight_candidates)
        )


def safe_save_model_for_hf_trainer(
    trainer: Trainer,
    output_dir: str,
    assert_adapter_checkpoint: bool = False,
):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir, _internal_call=True)
        if assert_adapter_checkpoint and trainer.args.should_save:
            assert_adapter_checkpoint_artifacts(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        if assert_adapter_checkpoint:
            trainer.model.save_pretrained(output_dir, state_dict=cpu_state_dict)
            torch.save(trainer.args, Path(output_dir) / "training_args.bin")
        else:
            trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
    if assert_adapter_checkpoint and trainer.args.should_save:
        assert_adapter_checkpoint_artifacts(output_dir)


class CheckpointFormatCallback(TrainerCallback):
    """This callback format checkpoint to make them standalone. For now, it copies all config
    files to /checkpoint-{step}/experiment_cfg/:
    - conf.yaml
    - initial_actions.npz
    - metadata.json
    """

    def __init__(
        self,
        run_name: str,
        exp_cfg_dir: Path | None = None,
        assert_adapter_checkpoint: bool = False,
    ):
        """
        Args:
            run_name: Name of the experiment run
            exp_cfg_dir: Path to the directory containing all experiment metadata
        """
        self.exp_cfg_dir = exp_cfg_dir
        self.assert_adapter_checkpoint = assert_adapter_checkpoint

    def on_save(self, args, state, control, **kwargs):
        """Called after the trainer saves a checkpoint."""
        if state.is_world_process_zero:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"

            # Copy experiment config directory if provided
            if self.exp_cfg_dir is not None:
                exp_cfg_dst = checkpoint_dir / self.exp_cfg_dir.name
                if self.exp_cfg_dir.exists():
                    shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)
            if self.assert_adapter_checkpoint:
                assert_adapter_checkpoint_artifacts(checkpoint_dir)
