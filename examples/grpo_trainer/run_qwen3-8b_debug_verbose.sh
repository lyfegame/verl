#!/bin/bash
# Debug version with extensive logging to identify hanging issue

set -x

# Set environment variables for debugging
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=300  # 5 minutes timeout to catch hangs faster
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

# Ray debugging
export RAY_DEDUP_LOGS=0
export RAY_memory_monitor_refresh_ms=0
export RAY_verbose_spill_logs=1
export RAY_BACKEND_LOG_LEVEL=debug

# PyTorch debugging
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=1  # This will make CUDA calls synchronous for better debugging

# Python debugging
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# Set Hydra to show full errors
export HYDRA_FULL_ERROR=1

echo "=== Starting VERL training with debug logging ==="
echo "Time: $(date)"
echo "Hostname: $(hostname)"
echo "GPUs available: $(nvidia-smi -L | wc -l)"
echo "Ray version: $(ray --version)"
echo "Python version: $(python3 --version)"
echo "PyTorch version: $(python3 -c 'import torch; print(torch.__version__)')"
echo "VLLM version: $(python3 -c 'import vllm; print(vllm.__version__)')"
echo "Tensordict version: $(python3 -c 'import tensordict; print(tensordict.__version__)')"

# Add Python path for better error messages
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create a wrapper Python script for additional debugging
cat > /tmp/debug_wrapper.py << 'EOF'
import sys
import os
import time
import traceback
import signal
import threading
from datetime import datetime

def log_with_timestamp(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {msg}", flush=True)

def signal_handler(signum, frame):
    log_with_timestamp(f"Received signal {signum}")
    traceback.print_stack(frame)
    sys.exit(1)

# Set up signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Periodic heartbeat to show the process is alive
def heartbeat():
    while True:
        log_with_timestamp("=== HEARTBEAT: Process is alive ===")
        time.sleep(30)

heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
heartbeat_thread.start()

# Import with logging
log_with_timestamp("Starting imports...")
try:
    import verl.trainer.main_ppo
    log_with_timestamp("Successfully imported verl.trainer.main_ppo")
except Exception as e:
    log_with_timestamp(f"Failed to import: {e}")
    traceback.print_exc()
    sys.exit(1)

# Monkey patch the trainer to add more logging
original_fit = verl.trainer.ppo.ray_trainer.RayPPOTrainer.fit

def logged_fit(self):
    log_with_timestamp("=== ENTERING fit() method ===")
    log_with_timestamp(f"Global steps: {getattr(self, 'global_steps', 'Not set')}")
    log_with_timestamp(f"Total training steps: {getattr(self, 'total_training_steps', 'Not set')}")
    
    # Patch the validation method
    original_validate = self._validate
    def logged_validate():
        log_with_timestamp("=== STARTING VALIDATION ===")
        result = original_validate()
        log_with_timestamp(f"=== VALIDATION COMPLETE, result keys: {list(result.keys()) if result else 'None'} ===")
        return result
    self._validate = logged_validate
    
    # Patch the timer context manager
    import verl.trainer.ppo.ray_trainer
    original_timer = verl.trainer.ppo.ray_trainer._timer
    def logged_timer(name, timing_raw):
        log_with_timestamp(f">>> Starting timer: {name}")
        with original_timer(name, timing_raw) as t:
            yield t
        log_with_timestamp(f"<<< Completed timer: {name} (took {timing_raw.get(name, 0):.2f}s)")
    verl.trainer.ppo.ray_trainer._timer = logged_timer
    
    try:
        log_with_timestamp("=== CALLING ORIGINAL fit() ===")
        return original_fit(self)
    except Exception as e:
        log_with_timestamp(f"=== ERROR IN fit(): {e} ===")
        traceback.print_exc()
        raise

verl.trainer.ppo.ray_trainer.RayPPOTrainer.fit = logged_fit

# Patch worker initialization
original_init_workers = verl.trainer.ppo.ray_trainer.RayPPOTrainer.init_workers

def logged_init_workers(self):
    log_with_timestamp("=== STARTING WORKER INITIALIZATION ===")
    log_with_timestamp(f"Resource pool manager: {self.resource_pool_manager}")
    log_with_timestamp(f"Hybrid engine: {self.hybrid_engine}")
    log_with_timestamp(f"Use critic: {self.use_critic}")
    log_with_timestamp(f"Use reference policy: {self.use_reference_policy}")
    log_with_timestamp(f"Use RM: {self.use_rm}")
    
    try:
        result = original_init_workers(self)
        log_with_timestamp("=== WORKER INITIALIZATION COMPLETE ===")
        return result
    except Exception as e:
        log_with_timestamp(f"=== ERROR IN WORKER INITIALIZATION: {e} ===")
        traceback.print_exc()
        raise

verl.trainer.ppo.ray_trainer.RayPPOTrainer.init_workers = logged_init_workers

# Run the main module
log_with_timestamp("=== STARTING MAIN MODULE ===")
sys.argv = ['verl.trainer.main_ppo'] + sys.argv[1:]
try:
    verl.trainer.main_ppo.main()
except Exception as e:
    log_with_timestamp(f"=== FATAL ERROR: {e} ===")
    traceback.print_exc()
    sys.exit(1)
EOF

# Run the training with the debug wrapper
python3 /tmp/debug_wrapper.py \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen3_8b_debug_verbose' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    ray_init.num_cpus=64 $@ 2>&1 | tee debug_output_$(date +%Y%m%d_%H%M%S).log 