#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
EXP_NAME="${EXP_NAME:-qwen3_math_baseline}"
TRAIN_FILES="${TRAIN_FILES:-['https://huggingface.co/datasets/EleutherAI/hendrycks_math?split=train']}"
VAL_FILES="${VAL_FILES:-['https://huggingface.co/datasets/EleutherAI/hendrycks_math?split=test','https://huggingface.co/datasets/openai/gsm8k?config=main&split=test','https://huggingface.co/datasets/JingzeShi/amc23?split=train','https://huggingface.co/datasets/math-ai/aime25?split=test','https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k?split=train']}"
LOG_DIR="${LOG_DIR:-outputs/${EXP_NAME}/sample_attention}"
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
TP_SIZE="${TP_SIZE:-1}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-4}"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length=512 \
  data.max_response_length=16384 \
  data.filter_overlong_prompts=True \
  data.truncation=left \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.do_sample=True \
  ++sample_attention.backward.enabled=true \
  ++sample_attention.backward.ema_decay=0.9 \
  ++sample_attention.backward.d_init_min=0.05 \
  ++sample_attention.backward.selection_ratio=1.0 \
  ++sample_attention.backward.min_selection_ratio=0.3 \
  ++sample_attention.backward.warmup_epochs=0 \
  ++sample_attention.backward.warmup_transition_epochs=0 \
  ++sample_attention.logging.log_dir="$LOG_DIR" \
  ++sample_attention.logging.use_kde=false \
  ++sample_attention.logging.log_math_categories=true \
  algorithm.use_kl_in_reward=False \
  algorithm.norm_adv_by_std_in_grpo=False \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name=lze \
  trainer.experiment_name="$EXP_NAME" \
  trainer.nnodes="$NNODES" \
  trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  "$@"