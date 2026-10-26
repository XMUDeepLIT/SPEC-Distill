# SPEC-Distill: Towards Efficient Speculative Distillation for Reinforcement Learning with Verifiable Rewards

<p align="center">
  Bingshuai Liu<sup>1,2*</sup>, Ante Wang<sup>1,2*</sup>, Zijun Min<sup>1,2*</sup>, Jiajun Cao<sup>1,2</sup><br>
  Haibo Zhang<sup>3</sup>, Liang Yao<sup>3</sup>, Shuman Liu<sup>3</sup>, Anxiang Zeng<sup>3</sup>, Jinsong Su<sup>1,2†</sup>
</p>

<p align="center">
  <sup>1</sup> School of Informatics, Xiamen University<br>
  <sup>2</sup> Key Laboratory of Digital Protection and Intelligent Processing of Intangible Cultural Heritage<br>
  of Fujian and Taiwan (Xiamen University), Ministry of Culture and Tourism, China<br>
  <sup>3</sup> Independent Researcher
</p>

<p align="center">
  <sup>*</sup> Equal contribution &nbsp; <sup>†</sup> Corresponding author<br>
  <a href="mailto:bsliu@stu.xmu.edu.cn">bsliu@stu.xmu.edu.cn</a> · <a href="mailto:jssu@xmu.edu.cn">jssu@xmu.edu.cn</a>
</p>

<p align="center">
  <a href="https://github.com/XMUDeepLIT/SPEC-Distill">Code</a> ·
  <a href="#-installation">Installation</a> ·
  <a href="#-training">Training</a> ·
  <a href="#-evaluation">Evaluation</a>
</p>

**SPEC-Distill moves teacher guidance offline, keeping online reinforcement learning student-only.** Built on [verl](https://github.com/volcengine/verl), it introduces teacher knowledge through speculative rollout construction and trains the student with verifiable rewards and GRPO.

## 💡 Overview

SPEC-Distill separates teacher guidance from the online training loop:

1. **Build an expert cache once.** Generate teacher drafts and token log-probabilities offline with Qwen3-14B. The same cache can support multiple student sizes.
2. **Warm up with teacher prefixes.** The student verifies cached drafts, retains the prefix before the first rejection, and generates the continuation.
3. **Continue with student self-reuse.** After the teacher-prefix phase, reuse refreshed student rollouts as drafts. Apply verifier rewards and the standard GRPO update throughout online training.

The provided configurations cover Qwen3-1.7B, 4B, and 8B students, alongside vanilla GRPO and Luffy baselines.

## 🧩 Installation

Use a CUDA environment compatible with **verl 0.5.x and vLLM 0.9.1**. A starting Docker image is:

```text
verlai/verl:app-verl0.5-vllm0.9.1-mcore0.12.2-te2.2
```

Alternatively, prepare the dependencies following the [verl 0.5.x installation guide](https://verl.readthedocs.io/en/v0.5.x/start/install.html#install-dependencies).

Download the code and install the additional dependencies inside your environment:

```bash
git clone https://github.com/XMUDeepLIT/SPEC-Distill.git
cd SPEC-Distill

pip install -r requirements.txt
pip install math-verify antlr4-python3-runtime==4.9.3

export PROJECT_DIR="$PWD"
export WORKING_DIR="$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
```

Run training from the repository root so Python uses the bundled `verl` implementation. This checkout does not include a root `setup.py` or `pyproject.toml`, so `pip install -e .` is not an installation step here.

The example launchers are configured for multi-GPU training, typically eight GPUs. Adjust GPU visibility, tensor parallelism, batch sizes, and memory utilization for your hardware.

**vLLM vocabulary patch:** Several launchers actively call `sudo bash training_scripts/fix_vllm_oov.sh`. Review this call before launching and comment it out if your environment does not need the vLLM 0.9.1 vocabulary patch.

## 🚀 Training

### 1. Configure model and output paths

```bash
export MODEL_PATH=/path/to/models
export SAVE_PATH=/path/to/checkpoints
export SPEC_DISTILL_EXPERT_DATA_DIR=/path/to/offline_teacher_rollouts
```

`MODEL_PATH` is a parent directory: the launchers load models from paths such as `$MODEL_PATH/Qwen3-1.7B` and `$MODEL_PATH/Qwen3-14B`. Download the corresponding model weights before running.

### 2. Prepare the training data

Datasets, generated teacher rollouts, and checkpoints are not bundled. The supplied non-thinking launchers expect this layout:

```text
data/
└── distill-data/
    ├── train_sample_no_think.parquet
    ├── train_sample_part_1_no_think.parquet
    ├── train_sample_part_2_no_think.parquet
    ├── train_sample_part_3_no_think.parquet
    ├── train_sample_part_4_no_think.parquet
    ├── train_sample_part_5_no_think.parquet
    ├── train_sample_part_6_no_think.parquet
    └── test.parquet
```

The full training file is used by student training; the six partitions are used by the offline teacher-generation launcher. Prepare these as verl-compatible parquet datasets, keeping the teacher inputs aligned with the student training examples. To use a different layout, edit `--dataset_name` and `--train_file_name` in the selected launcher. The shared backends read `data/<dataset_name>/<train_file_name>.parquet` and `data/<dataset_name>/test.parquet`.

### 3. Generate and register offline teacher rollouts

```bash
bash training_scripts/distill-data-generation/qwen-3-distill-14b-no-think.sh
```

The launcher runs Qwen3-14B over the six training partitions. Before student training, organize the generated rollout artifacts into the directory configured for `14b-no-think-16k` in [the expert registry](training_scripts/expert_registry.yaml). The registry resolves paths relative to `SPEC_DISTILL_EXPERT_DATA_DIR`; update its `path` field to match your cache location.

Check the available entries and resolved path:

```bash
python3 training_scripts/resolve_expert.py --list
python3 training_scripts/resolve_expert.py 14b-no-think-16k
```

The cache-linking helper expects paired `.jsonl` and `.pt` files with stems `-1`, `-2`, `-3`, `-4`, `-5`, and `0`. Ensure these artifacts are present before training. At launch, [init_expert_files.sh](training_scripts/init_expert_files.sh) links them into `rollouts/<run_name>/`.

### 4. Launch SPEC-Distill

Choose the launcher for your student model:

```bash
# Qwen3-1.7B
bash training_scripts/spec-distill/1.7B-think-14B-non-think.sh

# Qwen3-4B
bash training_scripts/spec-distill/4B-think-14B-non-think.sh

# Qwen3-8B
bash training_scripts/spec-distill/8B-think-14B-non-think.sh
```

Despite the `think` segment in the filenames, these launchers select the `14b-no-think-16k` teacher cache. Their shared backend is [train_grpo-spec-distill-warm-up.sh](training_scripts/train_grpo-spec-distill-warm-up.sh).

The main SPEC-Distill options are:

- `--expert_id`: select a teacher cache from the registry.
- `--spec_decoding True`: enable speculative prefix reuse.
- `--spec_warmup_epoch 1`: use teacher prefixes for the initial epoch.
- `--warmup_bias 4.0`: set the teacher-prefix log-lenience, corresponding to a lenience factor of `exp(4.0)`.
- `--bias 0.5`: set the subsequent speculative-phase log-lenience, corresponding to `exp(0.5)`.

### 5. Run baselines

```bash
# Vanilla GRPO; 4B and 8B launchers are in the same directory
bash training_scripts/vanilla-grpo/1.7B-grpo-non-think.sh

# Luffy; a 4B launcher is also provided
bash training_scripts/luffy/1.7B-luffy-deepmath-non-think-qwen3-14b.sh
```

Review the dataset and teacher-cache settings in each baseline launcher before running.

### Logging

Training defaults to console logging. To enable Weights & Biases, run `wandb login` and change the logger override in the relevant shared training backend to:

```text
trainer.logger=['console','wandb']
```

Set `PROJECT_NAME` in the selected launcher to organize your runs.

## 🧠 Evaluation

The math evaluation harness follows the [SPEC-RL](https://github.com/ShopeeLLM/Spec-RL) / [SimpleRL-Reason](https://github.com/hkust-nlp/simpleRL-reason) setup.

Configure your checkpoint run and base model, then launch:

```bash
export CHECKPOINT_DIR=/path/to/checkpoint_run
export INIT_MODEL=/path/to/models/Qwen3-1.7B

bash training_scripts/eval/start_spec_distill.sh
```

The checkpoint run directory should contain `global_step_*/actor/huggingface` subdirectories. The launcher evaluates:

- MATH500, GSM8K, OlympiadBench, and MMLU-STEM with one sample per problem.
- AIME24 and AIME25 with 32 samples per problem.
- AMC23 with 16 samples per problem.

Results are written under the checkpoint run directory in `eval_results`, `eval_results_avg32`, and `eval_results_avg16`.

AIME25 data is not included. Add `evaluation/simplelr_math_eval/data/aime25/test.jsonl`, or remove `aime25` from the benchmark list in the launcher.

The evaluation backend installs its own dependencies, including a different ANTLR runtime version from training. Use a separate evaluation environment to keep the training environment consistent. See [the evaluation README](evaluation/simplelr_math_eval/README.md) for details of the underlying harness.

## 📂 Repository Structure

```text
├── verl/                       # Training and speculative rollout implementation
├── training_scripts/
│   ├── spec-distill/            # SPEC-Distill launchers
│   ├── distill-data-generation/ # Offline teacher rollout generation
│   ├── vanilla-grpo/            # GRPO baselines
│   ├── luffy/                   # Luffy baselines
│   ├── eval/                    # Evaluation launchers
│   └── expert_registry.yaml    # Teacher-cache registry
├── custom_reward/              # Math verifier reward functions
├── evaluation/simplelr_math_eval/
├── fix_oov/                    # vLLM vocabulary patch helpers
└── data/                       # Location for prepared datasets
```

## 🙏 Acknowledgements

We thank the developers of [verl](https://github.com/volcengine/verl) for the RL training framework, [SPEC-RL](https://github.com/ShopeeLLM/Spec-RL) for its work on speculative rollouts, and [SimpleRL-Reason](https://github.com/hkust-nlp/simpleRL-reason) for the math evaluation harness. Our implementation also builds on [vLLM](https://github.com/vllm-project/vllm) for inference and [Qwen3](https://github.com/QwenLM/Qwen3) as the model family.
