#!/bin/bash
set -x
conda activate verl
export HYDRA_FULL_ERROR=1

# Enhanced debugging
echo "========================================" 
echo "Training started at: $(date)"
echo "========================================" 

# NCCL configuration
export NCCL_TIMEOUT=1800
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN  # Less verbose than INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

# Memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.7
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0

# Configuration
nnodes=1
nproc_per_node=8

# OPTIMIZED SETTINGS FOR STABILITY
lr=2e-4
micro_batch_size_per_gpu=1  # Increased from 1 to 2
train_batch_size=128  # Reduced from 128 to 64
max_length=22000  # Reduced from 22000 to 16384

# Validate batch size configuration
if [ $(($train_batch_size / $micro_batch_size_per_gpu % $nproc_per_node)) -ne 0 ]; then
    echo "Error: Invalid batch configuration"
    exit 1
fi

num_micro_batches=$((train_batch_size / micro_batch_size_per_gpu))
gradient_accumulation_steps=$((num_micro_batches / nproc_per_node))

echo "Configuration Summary:"
echo "  Micro batch size per GPU: $micro_batch_size_per_gpu"
echo "  Train batch size: $train_batch_size"
echo "  Gradient accumulation steps: $gradient_accumulation_steps"
echo "  Max sequence length: $max_length"
echo "  Number of micro batches: $num_micro_batches"

# Paths
DISK_ROOT=/home/tianhangzhu/gcs_view/home/tianhangzhu
ROOT_DIR=$DISK_ROOT/codev4/verlrollv2
export PYTHONPATH=$ROOT_DIR/verl:$PYTHONPATH

# Model and data
initial_model_path=Qwen/Qwen2.5-Coder-14B-Instruct
experiment_name=leadermodel14b_sft16k-lr-${lr}-trainbatch-${train_batch_size}-optimized-$(date +%Y%m%d-%H%M%S)

train_file=/home/tianhangzhu/data/test/leader_train_multiturn_10xsub35xother_optimized.jsonl
test_file=/home/tianhangzhu/data/test/leader_test_multiturn_10xsub35xother_optimized.jsonl

# Create output directories
output_dir=$DISK_ROOT/output_new/$experiment_name
log_dir=$DISK_ROOT/logs/$experiment_name
mkdir -p $output_dir
mkdir -p $log_dir
log_file="$log_dir/log.txt"

# Write initial info
echo "Host: $(hostname -I | awk '{print $1}')" > $log_file
echo "Start time: $(date)" >> $log_file
echo "Configuration:" >> $log_file
echo "  Model: $initial_model_path" >> $log_file
echo "  Max length: $max_length" >> $log_file
echo "  Train batch: $train_batch_size" >> $log_file
echo "  Micro batch: $micro_batch_size_per_gpu" >> $log_file
echo "  Grad accum: $gradient_accumulation_steps" >> $log_file

# GPU memory monitor function
monitor_gpu() {
    while true; do
        nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv >> "$log_dir/gpu_monitor.log" 2>&1
        sleep 30
    done
}

# Start GPU monitor
monitor_gpu &
MONITOR_PID=$!
echo "GPU monitor PID: $MONITOR_PID" >> $log_file

# Cleanup function
cleanup() {
    echo "Cleaning up..." >> $log_file
    kill $MONITOR_PID 2>/dev/null || true
    echo "End time: $(date)" >> $log_file
}
trap cleanup EXIT

# Run training with optimized settings
echo "Starting torchrun..." >> $log_file

torchrun --standalone \
    --nnodes=$nnodes \
    --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$train_file \
    data.val_files=$test_file \
    data.max_length=$max_length \
    data.truncation=right \
    optim.lr=$lr \
    optim.weight_decay=0.01 \
    optim.warmup_ratio=0.1 \
    data.micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
    data.train_batch_size=$train_batch_size \
    model.partial_pretrain=$initial_model_path \
    model.use_liger=True \
    model.enable_gradient_checkpointing=True \
    trainer.total_epochs=2 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.log_freq=10 \
    trainer.seed=42 \
    trainer.default_local_dir=$output_dir \
    trainer.project_name=leadersft_optimized \
    trainer.experiment_name=$experiment_name \
    "trainer.logger=['console','wandb']" \
    trainer.resume_mode="auto" \
    trainer.use_sudo_for_checkpoint=True \
    ulysses_sequence_parallel_size=8 \
    use_remove_padding=true \
    2>&1 | tee -a $log_file

echo "Training completed with exit code: $?" >> $log_file
