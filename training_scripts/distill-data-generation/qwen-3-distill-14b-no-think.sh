#!/bin/bash

pip install math-verify antlr4-python3-runtime==4.9.3


PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
MODEL_PATH=${MODEL_PATH:-/path/to/models}
SAVE_PATH=${SAVE_PATH:-/path/to/checkpoints}
PROJECT_NAME=spec-distill-data

# ray stop --force

# ray start --head \
# --num-gpus 8 \
# --dashboard-host 0.0.0.0 \
# --include-dashboard true

# for vllm 0.9.1: uncomment to fix oov issue
sudo bash ${PROJECT_DIR}/training_scripts/fix_vllm_oov.sh ${PROJECT_DIR}

cd ${PROJECT_DIR}
export WORKING_DIR=${PROJECT_DIR}


bash ${PROJECT_DIR}/training_scripts/train_grpo-expert-generation.sh \
    --dataset_name distill-data \
    --train_file_name train_sample_part_1_no_think \
    --model_name Qwen3-14B \
    --model_path ${MODEL_PATH} \
    --max_response_length 16384 \
    --train_batch_size 1024 \
    --rollout_n 8 \
    --rollout_gpu_memory_util 0.6 \
    --max_num_batched_tokens 32768 \
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
    --train_file_name train_sample_part_2_no_think \
    --model_name Qwen3-14B \
    --model_path ${MODEL_PATH} \
    --max_response_length 16384 \
    --train_batch_size 1024 \
    --rollout_n 8 \
    --rollout_gpu_memory_util 0.8 \
    --max_num_batched_tokens 32768 \
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
    --train_file_name train_sample_part_3_no_think \
    --model_name Qwen3-14B \
    --model_path ${MODEL_PATH} \
    --max_response_length 16384 \
    --train_batch_size 1024 \
    --rollout_n 8 \
    --rollout_gpu_memory_util 0.8 \
    --max_num_batched_tokens 32768 \
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
    --train_file_name train_sample_part_4_no_think \
    --model_name Qwen3-14B \
    --model_path ${MODEL_PATH} \
    --max_response_length 16384 \
    --train_batch_size 1024 \
    --rollout_n 8 \
    --rollout_gpu_memory_util 0.8 \
    --max_num_batched_tokens 32768 \
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
    --train_file_name train_sample_part_5_no_think \
    --model_name Qwen3-14B \
    --model_path ${MODEL_PATH} \
    --max_response_length 16384 \
    --train_batch_size 1024 \
    --rollout_n 8 \
    --rollout_gpu_memory_util 0.8 \
    --max_num_batched_tokens 32768 \
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
    --train_file_name train_sample_part_6_no_think \
    --model_name Qwen3-14B \
    --model_path ${MODEL_PATH} \
    --max_response_length 16384 \
    --train_batch_size 1024 \
    --rollout_n 8 \
    --rollout_gpu_memory_util 0.8 \
    --max_num_batched_tokens 32768 \
    --rollout_tp 8 \
    --total_epochs 20 \
    --total_steps 1 \
    --rollout_name vllm \
    --save_freq 10 \
    --checkpoint_path ${SAVE_PATH} \
    --project_name ${PROJECT_NAME}