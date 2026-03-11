import os
import torch
import gr00t

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import Gr00tPolicy

# change the following paths
MODEL_PATH = "nvidia/GR00T-N1.5-3B"

# REPO_PATH is the path of the pip install gr00t repo and one level up
REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
DATASET_PATH = os.path.join(REPO_PATH, "dyana_data")
# For pre-finetune baseline inference with the base model, use a built-in tag.
# After finetuning, switch this to the tag used in training (e.g. "new_embodiment")
# and point MODEL_PATH to your finetuned checkpoint directory.
EMBODIMENT_TAG = "gr1"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Pretrained Model
from gr00t.experiment.data_config import DATA_CONFIG_MAP

data_config = DATA_CONFIG_MAP["dyana_lora_11f_18d"]
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

# Align normalization/statistics and modality dimensions with the current dataset.
# This is useful for running baseline inference on custom data before finetuning.
policy.modality_transform.set_metadata(dataset.metadata)
policy.metadata = dataset.metadata

# Visualize one example data

step_data = dataset[0]

print(step_data)

print("\n\n ====================================")
for key, value in step_data.items():
    if isinstance(value, np.ndarray):
        print(key, value.shape)
    else:
        print(key, value)

# Run the policy
predicted_action = policy.get_action(step_data)
for key, value in predicted_action.items():
    print(key, value.shape)
    
