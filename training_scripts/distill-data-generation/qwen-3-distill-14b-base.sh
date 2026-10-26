#!/bin/bash

PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
MODEL_PATH=${MODEL_PATH:-/path/to/models}
SAVE_PATH=${SAVE_PATH:-/path/to/checkpoints}
PROJECT_NAME=spec-distill


# for vllm 0.9.1: uncomment to fix oov issue
sudo bash ${PROJECT_DIR}/training_scripts/fix_vllm_oov.sh ${PROJECT_DIR}

cd ${PROJECT_DIR}
export WORKING_DIR=${PROJECT_DIR}


bash ${PROJECT_DIR}/training_scripts/train_grpo-expert-generation.sh \
    --dataset_name distill-data \
    --train_file_name train_sample_part_1 \
    --model_name Qwen3-14B-Base \
    --model_path ${MODEL_PATH} \
    --max_response_length 4096 \
    --train_batch_size 1024 \
    --rollout_n 8 \
    --rollout_gpu_memory_util 0.8 \
    --rollout_tp 8 \
    --total_epochs 20 \
    --total_steps 1 \
    --rollout_name vllm \
    --save_freq 10 \
    --checkpoint_path ${SAVE_PATH} \
    --project_name ${PROJECT_NAME}

sleep 20

bash ${PROJECT_DIR}/training_scripts/train_grpo-expert-generation.sh \
    --dataset_name distill-data \
    --train_file_name train_sample_part_2 \
    --model_name Qwen3-14B-Base \
    --model_path ${MODEL_PATH} \
    --max_response_length 4096 \
    --train_batch_size 1024 \
    --rollout_n 8 \
    --rollout_gpu_memory_util 0.8 \
    --rollout_tp 4 \
    --total_epochs 20 \
    --total_steps 1 \
    --rollout_name vllm \
    --save_freq 10 \
    --checkpoint_path ${SAVE_PATH} \
    --project_name ${PROJECT_NAME}

sleep 20

bash ${PROJECT_DIR}/training_scripts/train_grpo-expert-generation.sh \
    --dataset_name distill-data \
    --train_file_name train_sample_part_3 \
    --model_name Qwen3-14B-Base \
    --model_path ${MODEL_PATH} \
    --max_response_length 4096 \
    --train_batch_size 1024 \
    --rollout_n 8 \
    --rollout_gpu_memory_util 0.8 \
    --rollout_tp 4 \
    --total_epochs 20 \
    --total_steps 1 \
    --rollout_name vllm \
    --save_freq 10 \
    --checkpoint_path ${SAVE_PATH} \
    --project_name ${PROJECT_NAME}

sleep 20

bash ${PROJECT_DIR}/training_scripts/train_grpo-expert-generation.sh \
    --dataset_name distill-data \
    --train_file_name train_sample_part_4 \
    --model_name Qwen3-14B-Base \
    --model_path ${MODEL_PATH} \
    --max_response_length 4096 \
    --train_batch_size 1024 \
    --rollout_n 8 \
    --rollout_gpu_memory_util 0.8 \
    --rollout_tp 4 \
    --total_epochs 20 \
    --total_steps 1 \
    --rollout_name vllm \
    --save_freq 10 \
    --checkpoint_path ${SAVE_PATH} \
    --project_name ${PROJECT_NAME}

sleep 20

bash ${PROJECT_DIR}/training_scripts/train_grpo-expert-generation.sh \
    --dataset_name distill-data \
    --train_file_name train_sample_part_5 \
    --model_name Qwen3-14B-Base \
    --model_path ${MODEL_PATH} \
    --max_response_length 4096 \
    --train_batch_size 1024 \
    --rollout_n 8 \
    --rollout_gpu_memory_util 0.8 \
    --rollout_tp 4 \
    --total_epochs 20 \
    --total_steps 1 \
    --rollout_name vllm \
    --save_freq 10 \
    --checkpoint_path ${SAVE_PATH} \
    --project_name ${PROJECT_NAME}

sleep 20

bash ${PROJECT_DIR}/training_scripts/train_grpo-expert-generation.sh \
    --dataset_name distill-data \
    --train_file_name train_sample_part_6 \
    --model_name Qwen3-14B-Base \
    --model_path ${MODEL_PATH} \
    --max_response_length 4096 \
    --train_batch_size 1024 \
    --rollout_n 8 \
    --rollout_gpu_memory_util 0.8 \
    --rollout_tp 4 \
    --total_epochs 20 \
    --total_steps 1 \
    --rollout_name vllm \
    --save_freq 10 \
    --checkpoint_path ${SAVE_PATH} \
    --project_name ${PROJECT_NAME}