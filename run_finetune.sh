#!/bin/bash

PROJECT=/data1/yfl_data/Dyana-GR00T
DATA=/data1/yfl_data/dyana_data
OUT=${PROJECT}/checkpoints/dyana_hand_task_lora_5k

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:1,2,4,5}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export LD_LIBRARY_PATH=/data1/yfl_data/miniconda3/envs/gr00t/lib/python3.10/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=${PROJECT}:${PYTHONPATH}

/data1/yfl_data/miniconda3/envs/gr00t/bin/python ${PROJECT}/scripts/gr00t_finetune.py \
  --dataset-path ${DATA} \
  --output-dir ${OUT} \
  --embodiment-tag dyana_hand_task \
  --video-backend decord \
  --num-gpus 4 \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --max-steps 5000 \
  --save-steps 500 \
  --learning-rate 5e-5 \
  --weight-decay 1e-5 \
  --warmup-ratio 0.05 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --report-to wandb
