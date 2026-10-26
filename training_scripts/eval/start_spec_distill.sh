#!/bin/bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
EVAL_SCRIPT=${EVAL_SCRIPT:-${PROJECT_DIR}/training_scripts/eval/eval_math_nodes.sh}

CHECKPOINT_DIR=${CHECKPOINT_DIR:-/path/to/checkpoint_run}
INIT_MODEL=${INIT_MODEL:-/path/to/base_model}
WANDB_PROJECT=${WANDB_PROJECT:-spec-distill}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

bash "$EVAL_SCRIPT" \
    --run_name "$CHECKPOINT_DIR" \
    --init_model "$INIT_MODEL" \
    --template qwen-boxed \
    --tp_size 1 \
    --add_step_0 false \
    --temperature 1.0 \
    --top_p 0.95 \
    --max_tokens 16000 \
    --benchmarks math500,gsm8k,olympiadbench,mmlu_stem \
    --output_dir eval_results \
    --n_sampling 1 \
    --wandb_project "$WANDB_PROJECT"

bash "$EVAL_SCRIPT" \
    --run_name "$CHECKPOINT_DIR" \
    --init_model "$INIT_MODEL" \
    --template qwen-boxed \
    --tp_size 1 \
    --add_step_0 false \
    --temperature 1.0 \
    --top_p 0.95 \
    --max_tokens 16000 \
    --benchmarks aime24,aime25 \
    --output_dir eval_results_avg32 \
    --n_sampling 32 \
    --wandb_project "$WANDB_PROJECT"

bash "$EVAL_SCRIPT" \
    --run_name "$CHECKPOINT_DIR" \
    --init_model "$INIT_MODEL" \
    --template qwen-boxed \
    --tp_size 1 \
    --add_step_0 false \
    --temperature 1.0 \
    --top_p 0.95 \
    --max_tokens 16000 \
    --benchmarks amc23 \
    --output_dir eval_results_avg16 \
    --n_sampling 16 \
    --wandb_project "$WANDB_PROJECT"
