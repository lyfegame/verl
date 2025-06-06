#!/bin/bash

# Test script for running Excel validation on a small dataset
# This tests the pipeline before running the full evaluation

echo "Testing Excel validation pipeline with Qwen3-8B on 5 tasks..."

# Add current directory to PYTHONPATH so Ray workers can find local modules
export PYTHONPATH=$(pwd):$PYTHONPATH
echo "PYTHONPATH set to: $PYTHONPATH"

set -x

# Run validation only for Excel tasks
python3 -m verl.trainer.main_ppo \
    +validation_only=true \
    data.custom_cls.path=verl/utils/dataset/excel_dataset.py \
    data.custom_cls.name=ExcelTaskDataset \
    data.train_files=/home/ubuntu/sharedusmidwest1/tianhangzhu/codev3/verltianhang/verl/test_excel_tasks_small.jsonl \
    data.val_files=/home/ubuntu/sharedusmidwest1/tianhangzhu/codev3/verltianhang/verl/test_excel_tasks_small.jsonl \
    data.train_batch_size=2 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    data.shuffle=false \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=false \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    critic.model.path=Qwen/Qwen3-8B \
    critic.model.trust_remote_code=true \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.use_remove_padding=true \
    critic.model.enable_gradient_checkpointing=false \
    critic.model.fsdp_config.param_offload=false \
    critic.model.fsdp_config.optimizer_offload=false \
    algorithm.use_kl_in_reward=false \
    custom_reward_function.path=verl/utils/reward/excel_reward.py \
    custom_reward_function.name=excel_comparison_reward \
    +custom_reward_function.reward_kwargs.reward_batch_size=2 \
    +custom_reward_function.reward_kwargs.max_workers=8 \
    reward_model.enable=false \
    trainer.critic_warmup=0 \
    trainer.logger=[console] \
    trainer.project_name=excel_validation_test \
    trainer.experiment_name=qwen3_8b_excel_test \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=true \
    hydra.run.dir=./outputs/excel_test_$(date +%Y%m%d_%H%M%S) $@ 