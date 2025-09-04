set -x
conda activate verl
export HYDRA_FULL_ERROR=1
nnodes=1
nproc_per_node=8
lr=1e-4
micro_batch_size_per_gpu=1
train_batch_size=32

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


########################################################
DISK_ROOT=/home/tianhangzhu/gcs_view/home/tianhangzhu
ROOT_DIR=$DISK_ROOT/codev4/verlrollv2

# Add the verl directory to the Python path
# This ensures that Python can find the verl module when running the script
export PYTHONPATH=$ROOT_DIR/verl:$PYTHONPATH
echo "Added $ROOT_DIR/verl to PYTHONPATH"
########################################################
# Create log directory
initial_model_path=Qwen/Qwen2.5-Coder-14B-Instruct

experiment_name=leadermodel14b_sft22k-Qwen2.5-Coder-14B-Instruct-lr-1e-4-trainbatch-32-ast_llmcompacted_1650convs-nodownsamplev2-20250818-024457

train_file=/home/tianhangzhu/gcs_view/home/tianhangzhu/data/leader_training_restast_llm_output_max_tokens_20000_1650convs/leader_train_8_nodownsample_otherupsample_atmost18x_spawn_subleader_append_subleader_result_upsample_5x.jsonl
test_file=/home/tianhangzhu/gcs_view/home/tianhangzhu/data/leader_training_restast_llm_output_max_tokens_20000_1650convs/leader_test_8_nodownsample_otherupsample_atmost18x_spawn_subleader_append_subleader_result_upsample_5x.jsonl
# train_file=/home/tianhangzhu/gcs_view/home/tianhangzhu/data/leader_training_restast_llm_output_max_tokens_20000_1650convs/leader_train_multiturn.jsonl
# test_file=/home/tianhangzhu/gcs_view/home/tianhangzhu/data/leader_training_restast_llm_output_max_tokens_20000_1650convs/leader_test_multiturn.jsonl
# NCCL timeout configuration
# Increase timeout to 30 minutes (1800 seconds) to handle long ALLTOALL operations
export NCCL_TIMEOUT=1800
# Alternative: You can also use NCCL_COMM_BLOCKING_WAIT_TIMEOUT (in seconds)
# export NCCL_COMM_BLOCKING_WAIT_TIMEOUT=1800

# Optional: Enable async error handling for better debugging
export NCCL_ASYNC_ERROR_HANDLING=1

# Optional: Increase NCCL debug level for more detailed logging
# export NCCL_DEBUG=INFO

max_length=22000
# initial_model_path=Qwen/Qwen2.5-14B-Instruct
mkdir -p $DISK_ROOT/logs/$experiment_name

log_file="$DISK_ROOT/logs/$experiment_name/log.txt"

# Write the IP address of the machine at the first line of the log file
hostname -I | awk '{print $1}' > $log_file


nohup torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$train_file \
    data.val_files=$test_file \
    data.max_length=$max_length \
    optim.lr=$lr \
    data.micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
    data.train_batch_size=$train_batch_size \
    model.partial_pretrain=$initial_model_path \
    model.use_liger=True \
    trainer.total_epochs=4 \
    model.enable_gradient_checkpointing=True \
    trainer.save_freq=30 \
    trainer.seed=42 \
    trainer.test_freq=30 \
    trainer.default_local_dir=$DISK_ROOT/output_new/$experiment_name \
    trainer.project_name=leadersftv4 \
    trainer.experiment_name=$experiment_name \
    trainer.logger=['console','wandb'] \
    trainer.resume_mode="auto" \
    trainer.use_sudo_for_checkpoint=True \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true $@ > >(tee -a $log_file) 2>&1 &
