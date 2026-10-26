#!/bin/bash
pip install math-verify antlr4-python3-runtime==4.9.3

PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
SAVE_PATH=${SAVE_PATH:-/path/to/checkpoints}
PROJECT_NAME=spec-distill


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# for vllm 0.9.1: uncomment to fix oov issue
sudo bash ${PROJECT_DIR}/training_scripts/fix_vllm_oov.sh ${PROJECT_DIR}

cd ${PROJECT_DIR}
export WORKING_DIR=${PROJECT_DIR}

MODEL_NAME=Qwen3-4B

MODEL_PATH=${MODEL_PATH:-/path/to/models}

bash ${PROJECT_DIR}/training_scripts/train_grpo-spec-distill-warm-up.sh \
    --expert_id 14b-no-think-16k \
    --dataset_name distill-data \
    --train_file_name train_sample_no_think \
    --model_name ${MODEL_NAME} \
    --model_path ${MODEL_PATH} \
    --max_response_length 16384 \
    --train_batch_size 1024 \
    --rollout_n 8 \
    --rollout_gpu_memory_util 0.85 \
    --rollout_tp 4 \
    --total_epochs 20 \
    --total_steps 102 \
    --rollout_name vllm \
    --save_freq 10 \
    --test_freq -1 \
    --spec_decoding True \
    --bias 0.5 \
    --spec_warmup_epoch 1 \
    --warmup_bias 4.0 \
    --checkpoint_path ${SAVE_PATH} \
    --project_name ${PROJECT_NAME} \
    --reward_fn_path ${PROJECT_DIR}/custom_reward/verl_math_verify.py \
    --suffix 14b-non-think
