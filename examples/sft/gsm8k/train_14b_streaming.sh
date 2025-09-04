#!/bin/bash
set -x
conda activate verl
export HYDRA_FULL_ERROR=1

# NCCL timeout configuration
export NCCL_TIMEOUT=1800
export NCCL_ASYNC_ERROR_HANDLING=1

nnodes=1
nproc_per_node=8
lr=2e-4
micro_batch_size_per_gpu=1
train_batch_size=128

# Validate batch size configuration
if [ $(($train_batch_size / $micro_batch_size_per_gpu % $nproc_per_node)) -ne 0 ]; then
    echo "Error: train_batch_size ($train_batch_size) divided by micro_batch_size_per_gpu ($micro_batch_size_per_gpu) must be divisible by nproc_per_node ($nproc_per_node)"
    exit 1
fi

num_micro_batches=$((train_batch_size / micro_batch_size_per_gpu))
echo "Number of micro batches per global step: $num_micro_batches"
echo "Number of micro batches per GPU: $((num_micro_batches / nproc_per_node))"

########################################################
DISK_ROOT=/home/tianhangzhu/gcs_view/home/tianhangzhu
ROOT_DIR=$DISK_ROOT/codev4/verlrollv2

export PYTHONPATH=$ROOT_DIR/verl:$PYTHONPATH
echo "Added $ROOT_DIR/verl to PYTHONPATH"
########################################################

initial_model_path=Qwen/Qwen2.5-Coder-14B-Instruct
experiment_name=leadermodel14b_sft22k-lr-${lr}-trainbatch-${train_batch_size}-streaming-dataset

train_file=/home/tianhangzhu/gcs_view/home/tianhangzhu/data/swegymrest_claude_sonnet_4_thinking_nonthinking_5settings_limit80_100default_v2/first_user_priority_output_max_tokens_20000_Noneconvs/leader_train_multiturn_10xsub35xother_optimized.jsonl
test_file=/home/tianhangzhu/gcs_view/home/tianhangzhu/data/swegymrest_claude_sonnet_4_thinking_nonthinking_5settings_limit80_100default_v2/first_user_priority_output_max_tokens_20000_Noneconvs/leader_test_multiturn_10xsub35xother_optimized.jsonl
max_length=22000

mkdir -p $DISK_ROOT/logs/$experiment_name
log_file="$DISK_ROOT/logs/$experiment_name/log.txt"

# Write the IP address of the machine at the first line of the log file
hostname -I | awk '{print $1}' > $log_file

# Add custom dataset configuration to use the streaming dataset
# The custom_cls configuration tells verl to use our custom dataset class
torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$train_file \
    data.val_files=$test_file \
    data.max_length=$max_length \
    data.custom_cls.path=verl.utils.dataset.streaming_sft_dataset \
    data.custom_cls.name=StreamingSFTDataset \
    optim.lr=$lr \
    data.micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
    data.train_batch_size=$train_batch_size \
    model.partial_pretrain=$initial_model_path \
    model.use_liger=True \
    trainer.total_epochs=4 \
    model.enable_gradient_checkpointing=True \
    trainer.save_freq=50 \
    trainer.seed=42 \
    trainer.test_freq=50 \
    trainer.default_local_dir=$DISK_ROOT/output_new/$experiment_name \
    trainer.project_name=leadersftv4_streaming \
    trainer.experiment_name=$experiment_name \
    trainer.logger=['console','wandb'] \
    trainer.resume_mode="auto" \
    trainer.use_sudo_for_checkpoint=True \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true $@ 2>&1 | tee -a $log_file
