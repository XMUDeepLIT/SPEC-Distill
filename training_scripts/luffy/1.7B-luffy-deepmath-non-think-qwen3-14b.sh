#!/bin/bash
# LUFFY training script on deepmath dataset.
# Aligned with: training_scripts/spec-distill/1.7B-think-14B-non-think.sh
# Difference: calls verl.mix_src.main_mix_ppo with LUFFY off-policy parameters.
pip install math-verify antlr4-python3-runtime==4.9.3

PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
MODEL_PATH=${MODEL_PATH:-/path/to/models}
SAVE_PATH=${SAVE_PATH:-/path/to/checkpoints}
PROJECT_NAME=luffy

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd ${PROJECT_DIR}
export WORKING_DIR=${PROJECT_DIR}

set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export RAY_memory_usage_threshold=0.99

DATASET_NAME=distill-data
TRAIN_FILE_NAME=train_sample_no_think_qwen3_14b
MODEL_NAME=Qwen3-1.7B
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=16384
TRAIN_BATCH_SIZE=1024
LEARNING_RATE=5e-7
PPO_MINI_BATCH_SIZE=256
PPO_MICRO_BATCH_SIZE=16
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=16
KL_LOSS_COEF=0.0001
KL_LOSS_TYPE="low_var_kl"
KL_COEF=0.001
ENTROPY_COEFF=0.001
ROLLOUT_N=8
ROLLOUT_GPU_MEMORY_UTIL=0.8
ROLLOUT_TP=2
TOTAL_EPOCHS=20
TOTAL_STEPS=102
SAVE_FREQ=10
TEST_FREQ=-1
NUM_GPU=8

DATA_PATH=${WORKING_DIR}/data
REWARD_FN_PATH=${WORKING_DIR}/custom_reward/verl_math_verify.py
LOG_PATH=${WORKING_DIR}/logs
mkdir -p $LOG_PATH

RUN_NAME="luffy_${DATASET_NAME}_${TRAIN_FILE_NAME}_${MODEL_NAME}"
LOG_FILE_PATH="$LOG_PATH/$RUN_NAME.log"
mkdir -p $SAVE_PATH/$RUN_NAME

python3 -m verl.mix_src.main_mix_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_PATH/$DATASET_NAME/$TRAIN_FILE_NAME.parquet \
    data.val_files=$DATA_PATH/$DATASET_NAME/test.parquet \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.shuffle=True \
    data.truncation='error' \
    +data.target_key=qwen3_14b \
    +data.max_target_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.model.path=$MODEL_PATH/$MODEL_NAME \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    custom_reward_function.path=$REWARD_FN_PATH \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=$NUM_GPU \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.val_before_train=False \
    trainer.default_local_dir=$SAVE_PATH/$RUN_NAME \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.total_epochs=$TOTAL_EPOCHS \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    +actor_rollout_ref.actor.checkpoint.keep_hf_model=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=18432 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=18432 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=18432 \
    +trainer.remove_previous_ckpt_in_save=True \
    +actor_rollout_ref.actor.use_off_policy_loss=True \
    +actor_rollout_ref.actor.off_policy_normalize=False \
    +actor_rollout_ref.actor.off_policy_reshape="p_div_p_0.1" \
    +actor_rollout_ref.actor.off_policy_loss_impl=token \
    +actor_rollout_ref.actor.off_policy_cliprange=0.2 \
    +actor_rollout_ref.actor.off_policy_max_clip=-1 \
    +actor_rollout_ref.actor.off_policy_min_clip=-1 \
    +actor_rollout_ref.actor.all_max_clip=-1 \
    +actor_rollout_ref.actor.clip_upper_bound=5.0 \
    +actor_rollout_ref.actor.on_policy_reshape=no_reshape \
    +actor_rollout_ref.actor.off_policy_reshape_weight=1.0 \
    +actor_rollout_ref.actor.off_policy_reshape_pow_exp=0.5 \
    +actor_rollout_ref.actor.on_policy_reshape_weight=1.0 \
    +actor_rollout_ref.actor.on_policy_reshape_pow_exp=0.5 \
    +actor_rollout_ref.actor.use_sft_multitask_loss=False \
    +actor_rollout_ref.actor.use_sft_prefix_reward=False \
    +actor_rollout_ref.rollout.prefix_share_across_samples=False \
    +actor_rollout_ref.rollout.prefix_strategy=random \
    +actor_rollout_ref.rollout.n_prefix=1 \
    +actor_rollout_ref.rollout.min_prefix_ratio=1.0 \
    +actor_rollout_ref.rollout.max_prefix_ratio=1.0 \
    +actor_rollout_ref.rollout.prefix_reward_weight_alpha=1.0 \
    +actor_rollout_ref.rollout.prefix_reward_weight_beta=1.0 \
    +actor_rollout_ref.rollout.prefix_steps=300 \
    +actor_rollout_ref.rollout.prefix_linear_max_ratio=0.8 2>&1 | tee -a $LOG_FILE_PATH
