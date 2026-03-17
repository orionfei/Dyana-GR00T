# Dyana-GR00T

LoRA fine-tuning and modular multimodal adaptation of **GR00T N1.5** for a dynamic grasping task with moving targets.

This project focuses on adapting a general robot foundation model to a harder setting: the robot must **observe a moving object first, infer its short-term trajectory, and then generate stable grasp actions**. The implementation is built on top of NVIDIA Isaac GR00T and extends it with a task-specific data pipeline, action-space adaptation, Unity evaluation, and a modular multimodal ablation framework.

## Highlights

- Adapted **GR00T N1.5** to a dynamic interaction task using **LoRA** while keeping the main backbone largely frozen.
- Reworked the control head from **32D to 18D** for the Dyana left-hand action/state space, with strict preflight checks to avoid silent shape mismatches.
- Designed an **11-frame observation / 10-step action** setup to better match the "observe-before-act" nature of moving-target grasping.
- Implemented a **modular multimodal pipeline** with three switchable components:
  - `motion_view`: frame-difference motion trail
  - `target_crop`: target-centric local crop
  - `motion_token`: compressed task semantics
- Built end-to-end tooling for **training, inference, Unity-based evaluation, and multimodal audit**, enabling clean ablation studies.
- Improved dynamic grasping success rate from **20% to 60%** in internal experiments.

## Task Overview

The task is a dynamic grasping scenario in which the robot must catch an object following one of several motion families:

- `linear`
- `circular`
- `harmonic`

Compared with static manipulation, this setting is harder because the model cannot rely on a single frame. It must reason over recent observations, infer short-term motion, and output stable future actions under uncertainty.

The current Dyana setup uses the following modalities:

- `video.ego_view`
- `state.left_hand` (18D)
- `action.left_hand` (18D)
- `annotation.human.action.task_description`

## Method

### 1. GR00T LoRA adaptation

The project fine-tunes GR00T with LoRA for task adaptation, with the default setup focusing on the action side of the model rather than fully tuning the entire backbone. This keeps training efficient while preserving the pretrained visual-language prior.

### 2. Task-specific temporal setup

The default Dyana configuration uses:

- **Observation horizon**: 11 frames
- **Action horizon**: 10 future steps
- **Action/state dimension**: 18D

This makes the model explicitly consume a short motion history instead of acting from a single image.

### 3. Modular multimodal enhancement

On top of the raw ego-view video, the project adds three optional modules for ablation:

- **Motion View**
  - derived from frame differencing
  - emphasizes the moving object's recent trajectory
- **Target Crop**
  - derived from motion localization
  - gives the model a more stable target-centric view
- **Motion Token**
  - compresses language annotations into `motion=linear|circular|harmonic`
  - removes unnecessary language variability and keeps only task-relevant motion semantics

These modules are implemented as independent transforms and can be enabled or disabled through `data_config`, which makes ablation experiments straightforward.

### 4. Evaluation and audit

The project provides:

- local inference for checkpoint sanity checks
- Unity-based closed-loop evaluation
- derived-view audit tools for:
  - `motion_map_empty_rate`
  - `target_crop_fallback_rate`
  - motion token distribution

## Repository Structure

```text
.
|-- gr00t/
|   |-- data/
|   |   `-- transform/
|   |       `-- dyana.py                 # motion view / target crop / motion token
|   `-- experiment/
|       `-- data_config.py              # modular Dyana configs
|-- scripts/
|   |-- gr00t_finetune.py               # main training entry
|   `-- audit_dyana_multimodal.py       # audit derived multimodal views
|-- eval_gr00t.py                       # Unity evaluation entry
|-- inference.py                        # local inference sanity check
|-- run_finetune.sh                     # convenience launcher
|-- dyana_data/                         # expected LeRobot-compatible dataset root
`-- Dyana.pdf                           # task description document
```

## Multimodal Ablation Configs

The following phase-1 configs are currently available:

| Config | Video Streams | Language | Use Case |
|---|---|---|---|
| `dyana_lora_11f_18d` | `ego_view` | raw task text | baseline |
| `dyana_motion_token_11f_18d` | `ego_view` | motion token | language compression only |
| `dyana_motion_view_11f_18d` | `ego_view + motion_view` | raw task text | motion trajectory emphasis |
| `dyana_motion_view_token_11f_18d` | `ego_view + motion_view` | motion token | motion view + language compression |
| `dyana_target_crop_11f_18d` | `ego_view + target_crop` | raw task text | target-centric visual focus |
| `dyana_target_crop_token_11f_18d` | `ego_view + target_crop` | motion token | crop + language compression |
| `dyana_motion_crop_11f_18d` | `ego_view + motion_view + target_crop` | raw task text | strongest visual enhancement |
| `dyana_motion_crop_token_11f_18d` | `ego_view + motion_view + target_crop` | motion token | full phase-1 setting |

## Installation

Recommended environment:

- Python 3.10
- CUDA-enabled PyTorch
- Linux for training and evaluation
- `decord` or another supported video backend

Typical setup:

```bash
conda create -n gr00t python=3.10
conda activate gr00t
pip install --upgrade setuptools
pip install -e .[base]
```

If you train with GPU attention kernels, install a compatible `flash-attn` version for your CUDA setup.

## Dataset Format

The project expects a **LeRobot-compatible dataset** under `dyana_data/`.

At minimum, the Dyana task uses:

- `video.ego_view`
- `state.left_hand`
- `action.left_hand`
- `annotation.human.action.task_description`

Notes:

- The full private dataset used for experiments is **not released** in this repository.
- Lightweight metadata or local examples may exist, but raw videos and full trajectories are not distributed here.

## Quick Start

### 1. Train the baseline

Use the convenience script:

```bash
DATA_CONFIG=dyana_lora_11f_18d bash run_finetune.sh
```

Or call the training entry directly:

```bash
python scripts/gr00t_finetune.py \
  --dataset-path ./dyana_data \
  --output-dir ./checkpoints/dyana_baseline \
  --data-config dyana_lora_11f_18d \
  --embodiment-tag dyana_hand_task \
  --video-backend decord \
  --num-gpus 1 \
  --batch-size 4 \
  --max-steps 5000 \
  --save-steps 500 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05
```

### 2. Train a multimodal variant

Example: full phase-1 setting with motion view, target crop, and motion token.

```bash
python scripts/gr00t_finetune.py \
  --dataset-path ./dyana_data \
  --output-dir ./checkpoints/dyana_motion_crop_token \
  --data-config dyana_motion_crop_token_11f_18d \
  --embodiment-tag dyana_hand_task \
  --video-backend decord \
  --num-gpus 1 \
  --batch-size 4 \
  --max-steps 5000 \
  --save-steps 500 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05
```

### 3. Audit derived multimodal inputs

```bash
python scripts/audit_dyana_multimodal.py \
  --dataset-path ./dyana_data \
  --data-config dyana_motion_crop_token_11f_18d \
  --video-backend decord \
  --sample-count 128 \
  --save-examples-dir ./audit_examples
```

This script reports:

- `motion_map_empty_rate`
- `target_crop_fallback_rate`
- `motion_token_counts`

### 4. Run local inference

```bash
set DYANA_GR00T_MODEL_PATH=./checkpoints/dyana_baseline/checkpoint-5000
set DYANA_GR00T_DATA_CONFIG=dyana_lora_11f_18d
python inference.py
```

On Linux/macOS, replace `set` with `export`.

### 5. Run Unity evaluation

```bash
python eval_gr00t.py \
  --model-path ./checkpoints/dyana_baseline/checkpoint-5000 \
  --data-config dyana_motion_crop_token_11f_18d \
  --dataset-path ./dyana_data \
  --traj-start 982 \
  --traj-end 1000 \
  --repeat 1 \
  --eval-window 10 \
  --output-dir ./evaluation_results
```

## Experimental Notes

- The project defaults to **LoRA on the action side** rather than full-model tuning.
- Strict dataset preflight checks are enabled in training to catch:
  - horizon mismatches
  - dimension mismatches
  - missing keys
  - incorrect multimodal config composition
- Video metadata resolution is synchronized against decoded frames to avoid stale dataset metadata breaking training.

## Current Results

Current internal results show:

- dynamic grasping success rate improved from **20% to 60%**
- more stable adaptation after **32D -> 18D** action-space remapping
- cleaner ablation workflow with switchable multimodal modules

The most useful evaluation split is by motion family:

- `straight`
- `circular`
- `harmonic`

## Roadmap

Phase-1 multimodal modules are implemented. The next planned step is a **structured motion-context state input**, for example:

- motion-type one-hot
- normalized motion parameters
- predicted intercept point
- timing bucket

This would extend the current visual-language enhancement into a stronger numeric state prior for dynamic interaction.

## Acknowledgements

This repository is built on top of:

- [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- [GR00T N1.5 model](https://huggingface.co/nvidia/GR00T-N1.5-3B)
- [Hugging Face LeRobot](https://github.com/huggingface/lerobot)

Thanks to the upstream projects for the model, data schema, and training infrastructure that made this task-specific adaptation possible.

## Citation

If you find this repository useful, please cite the upstream GR00T work and reference this project as a task-specific adaptation based on NVIDIA Isaac GR00T.
