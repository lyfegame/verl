#!/bin/bash
set -x

# Remove debug overhead for 10-20% speedup
unset NCCL_DEBUG  

# Set C compiler to fix "Failed to find C compiler" error
export CC=/usr/bin/gcc

# Enable expandable segments for CUDA memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nproc_per_node=8

# Training configuration
lr=5e-4
micro_batch_size_per_gpu=1
train_batch_size=32
max_length=16384

# Validate that train_batch_size is divisible by (micro_batch_size_per_gpu * nproc_per_node)
# This ensures that the number of micro batches per global step is an integer
# and can be evenly distributed across all GPUs
if [ $(($train_batch_size / $micro_batch_size_per_gpu % $nproc_per_node)) -ne 0 ]; then
    echo "Error: train_batch_size ($train_batch_size) divided by micro_batch_size_per_gpu ($micro_batch_size_per_gpu) must be divisible by nproc_per_node ($nproc_per_node)"
    echo "This ensures that micro batches can be evenly distributed across all GPUs"
    exit 1
fi

# Calculate and display the number of micro batches per global step
num_micro_batches=$((train_batch_size / micro_batch_size_per_gpu))
echo "Number of micro batches per global step: $num_micro_batches"
echo "Number of micro batches per GPU: $((num_micro_batches / nproc_per_node))"

# Dataset paths
train_file=/mnt/shared-nfs/home/tianhangzhu/data/leader_training_17thjuly_swe_gym_train_rest/tokenized_stepwise_context_compacted_16k_ast_llm_output_max_tokens_16000/leader_train_multiturn.jsonl
test_file=/mnt/shared-nfs/home/tianhangzhu/data/leader_training_17thjuly_swe_gym_train_rest/tokenized_stepwise_context_compacted_16k_ast_llm_output_max_tokens_16000/leader_test_multiturn.jsonl

# Model configuration
initial_model_path=Qwen/Qwen2.5-Coder-7B-Instruct

# Experiment naming
experiment_name=leadermodel_sft16k-$(basename ${initial_model_path})-lr-${lr}-trainbatch-${train_batch_size}-ast_llm_output_multiturn_codev4-$(date +%Y%m%d-%H%M%S)
save_path=/home/tianhangzhu/output/$experiment_name

# Create log directory
mkdir -p $save_path/logs
log_file="$save_path/logs/log_$experiment_name.txt"

# Write the IP address of the machine at the first line of the log file
hostname -I | awk '{print $1}' > $log_file

echo "Starting training with the following configuration:" | tee -a $log_file
echo "  Model: $initial_model_path" | tee -a $log_file
echo "  Train file: $train_file" | tee -a $log_file
echo "  Test file: $test_file" | tee -a $log_file
echo "  Learning rate: $lr" | tee -a $log_file
echo "  Train batch size: $train_batch_size" | tee -a $log_file
echo "  Micro batch size per GPU: $micro_batch_size_per_gpu" | tee -a $log_file
echo "  Max length: $max_length" | tee -a $log_file
echo "  Number of GPUs: $nproc_per_node" | tee -a $log_file

# Run the training
torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$train_file \
    data.val_files=$test_file \
    data.train_batch_size=$train_batch_size \
    data.micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
    data.max_length=$max_length \
    data.custom_cls.path=verl.utils.dataset.sft_dataset \
    data.custom_cls.name=SFTDataset \
    optim.lr=$lr \
    model.partial_pretrain=$initial_model_path \
    model.enable_gradient_checkpointing=True \
    model.use_liger=True \
    model.eval_interval=50 \
    model.save_interval=50 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=leadermodel-sft-experiment \
    trainer.experiment_name=$experiment_name \
    trainer.total_epochs=10 \
    trainer.logger=['console','wandb'] \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true \
    trainer.default_hdfs_dir=null $@ 2>&1 | tee -a $log_file

echo "Training completed. Check logs at $log_file" | tee -a $log_file