#!/bin/bash
# Simple debug version with Python logging

set -x

# Set environment variables for debugging
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=300  # 5 minutes timeout
export NCCL_ASYNC_ERROR_HANDLING=1

# Ray debugging
export RAY_DEDUP_LOGS=0

# Python debugging
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

# Enable Python logging for VERL modules
export VERL_LOG_LEVEL=DEBUG
export TRANSFORMERS_VERBOSITY=info

# Add verbose logging for specific components
export CUDA_LAUNCH_BLOCKING=1  # Makes CUDA synchronous for better debugging

echo "=== Environment Setup ==="
echo "Time: $(date)"
echo "Hostname: $(hostname)"
echo "GPU count: $(nvidia-smi -L | wc -l)"
echo "Python: $(python3 --version)"

# Run with Python verbose logging
python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen3_8b_simple_debug' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=0 \
    trainer.test_freq=0 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    ray_init.num_cpus=32 $@ 2>&1 | tee simple_debug_$(date +%Y%m%d_%H%M%S).log 