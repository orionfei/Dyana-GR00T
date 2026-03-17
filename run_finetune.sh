#!/bin/bash

PROJECT=/data1/yfl_data/Dyana-GR00T
DATA=/data1/yfl_data/dyana_data
DATA_CONFIG=${DATA_CONFIG:-dyana_lora_11f_18d}
RUN_NAME=${RUN_NAME:-dyana_hand_task_${DATA_CONFIG}_5k}
OUT=${OUT:-${PROJECT}/checkpoints/${RUN_NAME}}
DYANA_OPTIMIZED=${DYANA_OPTIMIZED:-0}
DYANA_STAGE=${DYANA_STAGE:-pilot}
DYANA_SUBSET_SEED=${DYANA_SUBSET_SEED:-42}
DYANA_TRAIN_EPISODES_PER_TASK=${DYANA_TRAIN_EPISODES_PER_TASK:-}
DYANA_HOLDOUT_EPISODES_PER_TASK=${DYANA_HOLDOUT_EPISODES_PER_TASK:-}
DYANA_BASE_INDEX_STRIDE=${DYANA_BASE_INDEX_STRIDE:-}
DYANA_TASK_REPEAT_LINEAR=${DYANA_TASK_REPEAT_LINEAR:-2}
DYANA_TASK_REPEAT_CIRCULAR=${DYANA_TASK_REPEAT_CIRCULAR:-1}
DYANA_TASK_REPEAT_HARMONIC=${DYANA_TASK_REPEAT_HARMONIC:-1}
DYANA_EVAL_STEPS=${DYANA_EVAL_STEPS:-500}
DYANA_EARLY_STOP_PATIENCE=${DYANA_EARLY_STOP_PATIENCE:-3}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,4,5}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export LD_LIBRARY_PATH=/data1/yfl_data/miniconda3/envs/gr00t/lib/python3.10/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=${PROJECT}:${PYTHONPATH}

echo "PROJECT=${PROJECT}"
echo "DATA=${DATA}"
echo "DATA_CONFIG=${DATA_CONFIG}"
echo "OUT=${OUT}"
echo "DYANA_OPTIMIZED=${DYANA_OPTIMIZED}"

EXTRA_ARGS=()
if [ "${DYANA_OPTIMIZED}" = "1" ]; then
  EXTRA_ARGS+=(
    --dyana-optimized
    --dyana-stage "${DYANA_STAGE}"
    --dyana-subset-seed "${DYANA_SUBSET_SEED}"
    --dyana-task-repeat-linear "${DYANA_TASK_REPEAT_LINEAR}"
    --dyana-task-repeat-circular "${DYANA_TASK_REPEAT_CIRCULAR}"
    --dyana-task-repeat-harmonic "${DYANA_TASK_REPEAT_HARMONIC}"
    --dyana-eval-steps "${DYANA_EVAL_STEPS}"
    --dyana-early-stop-patience "${DYANA_EARLY_STOP_PATIENCE}"
  )
  if [ -n "${DYANA_TRAIN_EPISODES_PER_TASK}" ]; then
    EXTRA_ARGS+=(--dyana-train-episodes-per-task "${DYANA_TRAIN_EPISODES_PER_TASK}")
  fi
  if [ -n "${DYANA_HOLDOUT_EPISODES_PER_TASK}" ]; then
    EXTRA_ARGS+=(--dyana-holdout-episodes-per-task "${DYANA_HOLDOUT_EPISODES_PER_TASK}")
  fi
  if [ -n "${DYANA_BASE_INDEX_STRIDE}" ]; then
    EXTRA_ARGS+=(--dyana-base-index-stride "${DYANA_BASE_INDEX_STRIDE}")
  fi
fi

TRAIN_ARGS=(
  --dataset-path ${DATA}
  --output-dir ${OUT}
  --data-config ${DATA_CONFIG}
  --embodiment-tag dyana_hand_task
  --video-backend decord
  --num-gpus 4
  --batch-size 4
  --save-steps 500
  --weight-decay 1e-5
  --lora-rank 32
  --lora-alpha 64
  --lora-dropout 0.05
  --report-to wandb
)

if [ "${DYANA_OPTIMIZED}" = "1" ]; then
  if [ -n "${GRADIENT_ACCUMULATION_STEPS:-}" ]; then
    TRAIN_ARGS+=(--gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}")
  fi
  if [ -n "${MAX_STEPS:-}" ]; then
    TRAIN_ARGS+=(--max-steps "${MAX_STEPS}")
  fi
  if [ -n "${LEARNING_RATE:-}" ]; then
    TRAIN_ARGS+=(--learning-rate "${LEARNING_RATE}")
  fi
  if [ -n "${WARMUP_RATIO:-}" ]; then
    TRAIN_ARGS+=(--warmup-ratio "${WARMUP_RATIO}")
  fi
else
  TRAIN_ARGS+=(
    --gradient-accumulation-steps 2
    --max-steps 5000
    --learning-rate 5e-5
    --warmup-ratio 0.05
  )
fi

/data1/yfl_data/miniconda3/envs/gr00t/bin/python ${PROJECT}/scripts/gr00t_finetune.py \
  "${TRAIN_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
