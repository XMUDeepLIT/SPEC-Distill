#!/bin/bash
pip install math-verify antlr4-python3-runtime==4.9.3

PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
MODEL_PATH=${MODEL_PATH:-/path/to/models}
SAVE_PATH=${SAVE_PATH:-/path/to/checkpoints}
PROJECT_NAME=spec-distill

# for vllm 0.9.1: uncomment to fix oov issue
sudo bash ${PROJECT_DIR}/training_scripts/fix_vllm_oov.sh ${PROJECT_DIR}

cd ${PROJECT_DIR}
export WORKING_DIR=${PROJECT_DIR}


bash ${PROJECT_DIR}/training_scripts/train_grpo.sh \
    --dataset_name distill-data \
    --train_file_name train_sample_no_think \
    --model_name Qwen3-4B \
    --model_path ${MODEL_PATH} \
    --max_response_length 16384 \
    --train_batch_size 1024 \
    --rollout_n 8 \
    --rollout_gpu_memory_util 0.85 \
    --rollout_tp 2 \
    --total_epochs 20 \
    --total_steps 102 \
    --rollout_name vllm \
    --save_freq 10 \
    --checkpoint_path ${SAVE_PATH} \
    --project_name ${PROJECT_NAME}