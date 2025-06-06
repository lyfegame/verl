#!/bin/bash

# Example script for running Excel validation using main_ppo.py
# This demonstrates how to use the validation_only mode

set -x

# Run validation only for Excel tasks
python3 -m verl.trainer.main_ppo \
    validation_only=true \
    data.use_excel_dataset=true \
    data.train_files=/home/ubuntu/sharedusmidwest1/tianhangzhu/codev3/prototype-new-infra_ourtv1/evaluation/tasks/rm_data/merged_test83_72_data_noread_addr.jsonl \
    data.val_files=/home/ubuntu/sharedusmidwest1/tianhangzhu/codev3/prototype-new-infra_ourtv1/evaluation/tasks/rm_data/merged_test83_72_data_noread_addr.jsonl \
    data.train_batch_size=32 \
    data.max_prompt_len=4096 \
    data.max_response_len=2048 \
    data.filter_overlong_prompts=true \
    data.truncation='error' \
    data.shuffle=false \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    algorithm.use_kl_in_reward=false \
    custom_reward_function.path=verl/utils/reward/excel_reward.py \
    custom_reward_function.name=excel_comparison_reward \
    custom_reward_function.reward_kwargs.reward_batch_size=32 \
    custom_reward_function.reward_kwargs.max_workers=8 \
    reward_model.enable=false \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='excel_validation' \
    trainer.experiment_name='excel_task_validation' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=true $@ 