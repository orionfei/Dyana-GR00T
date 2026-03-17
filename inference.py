import os
from pathlib import Path

import torch

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.transform.video import VideoCrop
from gr00t.model.policy import Gr00tPolicy

# change the following paths
REPO_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(REPO_PATH, "dyana_data")
DEFAULT_MODEL_ROOT = Path(REPO_PATH) / "checkpoints" / "dyana_hand_task_lora"
# This must match the embodiment tag used during finetuning.
EMBODIMENT_TAG = "dyana_hand_task"

device = "cuda" if torch.cuda.is_available() else "cpu"


def resolve_model_path() -> str:
    env_model_path = os.environ.get("DYANA_GR00T_MODEL_PATH")
    if env_model_path:
        return env_model_path

    checkpoint_dirs = sorted(DEFAULT_MODEL_ROOT.glob("checkpoint-*"))
    if checkpoint_dirs:
        return str(checkpoint_dirs[-1])
    return str(DEFAULT_MODEL_ROOT)


MODEL_PATH = resolve_model_path()

# Load Pretrained Model
from gr00t.experiment.data_config import DATA_CONFIG_MAP

DATA_CONFIG_NAME = os.environ.get("DYANA_GR00T_DATA_CONFIG", "dyana_lora_11f_18d")
assert DATA_CONFIG_NAME in DATA_CONFIG_MAP, (
    f"Unknown data config {DATA_CONFIG_NAME}. Available: {sorted(DATA_CONFIG_MAP.keys())}"
)
print(f"Using data config: {DATA_CONFIG_NAME}")

data_config = DATA_CONFIG_MAP[DATA_CONFIG_NAME]
modality_config = data_config.modality_config()
modality_transform = data_config.transform()

policy = Gr00tPolicy(
    model_path=MODEL_PATH,
    embodiment_tag=EMBODIMENT_TAG,
    modality_config=modality_config,
    modality_transform=modality_transform,
    device=device,
)

# print out the policy model architecture
# print(policy.model)

# Load Dataset
import numpy as np

modality_config = policy.modality_config

# print(modality_config.keys())

# for key, value in modality_config.items():
#     if isinstance(value, np.ndarray):
#         print(key, value.shape)
#     else:
#         print(key, value)

# Create the dataset
dataset = LeRobotSingleDataset(
    dataset_path=DATASET_PATH,
    modality_configs=modality_config,
    video_backend="decord",
    video_backend_kwargs=None,
    transforms=None,  # We'll handle transforms separately through the policy
    embodiment_tag=EMBODIMENT_TAG,
)

# Read one raw sample first, then align metadata with the true video resolution.
step_data = dataset[0]
for video_key in modality_config["video"].modality_keys:
    if video_key not in step_data:
        continue
    # step_data[video_key] shape: [T, H, W, C]
    frames = step_data[video_key]
    height, width = int(frames.shape[-3]), int(frames.shape[-2])
    sub_key = video_key.split(".", 1)[1]
    if sub_key in dataset.metadata.modalities.video:
        dataset.metadata.modalities.video[sub_key].resolution = (width, height)
        print(f"Updated metadata resolution for {video_key}: {(width, height)}")

# Align normalization/statistics and modality dimensions with the current dataset.
# `Gr00tPolicy` first loads checkpoint metadata. If that metadata has a different
# camera resolution, VideoCrop may keep stale (height, width). Reset it so current
# dataset metadata is always used.
for transform in policy.modality_transform.transforms:
    if isinstance(transform, VideoCrop):
        transform.height = None
        transform.width = None

policy.modality_transform.set_metadata(dataset.metadata)
policy.metadata = dataset.metadata

# Visualize one example data

# print(step_data)

print("\n\n ====================================")
# for key, value in step_data.items():
#     if isinstance(value, np.ndarray):
#         print(key, value.shape)
#     else:
#         print(key, value)

# Run the policy
predicted_action = policy.get_action(step_data)

target_action_key = modality_config["action"].modality_keys[0]
assert target_action_key in predicted_action, (
    f"Expected action key {target_action_key}, got {list(predicted_action.keys())}"
)
target_action = predicted_action[target_action_key]
assert target_action.shape[-1] == 18, (
    f"Expected 18-dim action for {target_action_key}, got shape {target_action.shape}"
)

for key, value in predicted_action.items():
    print(key, value.shape)
    
