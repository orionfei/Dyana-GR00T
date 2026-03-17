#!/bin/bash

PROJECT=${PROJECT:-/data1/yfl_data/Dyana-GR00T}
DATA=${DATA:-/data1/yfl_data/dyana_data}
DATA_CONFIG=${DATA_CONFIG:-dyana_motion_crop_token_11f_18d}
RUN_NAME=${RUN_NAME:-awbc_round0}
INIT_CHECKPOINT=${INIT_CHECKPOINT:-${PROJECT}/checkpoints/dyana_hand_task_${DATA_CONFIG}_5k}
ROLLOUT_ROOT=${ROLLOUT_ROOT:-${PROJECT}/rollouts}
ROLLOUT_DIR=${ROLLOUT_DIR:-${ROLLOUT_ROOT}/${RUN_NAME}}
OUT=${OUT:-${PROJECT}/checkpoints/${RUN_NAME}}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export LD_LIBRARY_PATH=/data1/yfl_data/miniconda3/envs/gr00t/lib/python3.10/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=${PROJECT}:${PYTHONPATH}

PYTHON_BIN=${PYTHON_BIN:-/data1/yfl_data/miniconda3/envs/gr00t/bin/python}
EPISODES_PER_TASK=${EPISODES_PER_TASK:-60}
NUM_GPUS=${NUM_GPUS:-1}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-1}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8765}

echo "PROJECT=${PROJECT}"
echo "DATA=${DATA}"
echo "DATA_CONFIG=${DATA_CONFIG}"
echo "INIT_CHECKPOINT=${INIT_CHECKPOINT}"
echo "ROLLOUT_DIR=${ROLLOUT_DIR}"
echo "OUT=${OUT}"
echo "Unity Inspector reminder: set enableStepFeedback=true before rollout collection."

${PYTHON_BIN} ${PROJECT}/scripts/collect_dyana_rl_rollouts.py \
  --model-path ${INIT_CHECKPOINT} \
  --dataset-path ${DATA} \
  --data-config ${DATA_CONFIG} \
  --output-dir ${ROLLOUT_ROOT} \
  --run-name ${RUN_NAME} \
  --episodes-per-task ${EPISODES_PER_TASK} \
  --host ${HOST} \
  --port ${PORT}

${PYTHON_BIN} ${PROJECT}/scripts/finetune_dyana_awbc.py \
  --demo-dataset ${DATA} \
  --rollout-dir ${ROLLOUT_DIR} \
  --init-checkpoint ${INIT_CHECKPOINT} \
  --output-dir ${OUT} \
  --data-config ${DATA_CONFIG} \
  --embodiment-tag dyana_hand_task \
  --video-backend decord \
  --num-gpus ${NUM_GPUS} \
  --batch-size ${BATCH_SIZE} \
  --gradient-accumulation-steps ${GRAD_ACCUM} \
  --learning-rate 5e-5 \
  --weight-decay 1e-5 \
  --warmup-ratio 0.05 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --report-to wandb
