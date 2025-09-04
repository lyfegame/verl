set -x
conda activate verl
export HYDRA_FULL_ERROR=1
nnodes=1
nproc_per_node=8
lr=5e-4
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
ROOT_DIR=/mnt/shared-nfs/home/tianhangzhu/codev4/verlrollv2

# Add the verl directory to the Python path
# This ensures that Python can find the verl module when running the script
export PYTHONPATH=$ROOT_DIR/verl:$PYTHONPATH
echo "Added $ROOT_DIR/verl to PYTHONPATH"
########################################################
# Create log directory
initial_model_path=Qwen/Qwen2.5-Coder-14B-Instruct

experiment_name=leadermodel14b_sft22k_8groups_balanced_atmost5x-$(basename ${initial_model_path})-lr-${lr}-trainbatch-${train_batch_size}-ast_llmcompacted_1650convs

train_file=/mnt/shared-nfs/home/tianhangzhu/data/leader_training_restast_llm_output_max_tokens_20000_1650convs_8_groups_balanced_atmost5x/leader_train_8_groups_balanced_atmost5x.jsonl
test_file=/mnt/shared-nfs/home/tianhangzhu/data/leader_training_restast_llm_output_max_tokens_20000_1650convs_8_groups_balanced_atmost5x/leader_test_8_groups_balanced_atmost5x.jsonl


max_length=22000
# initial_model_path=Qwen/Qwen2.5-14B-Instruct
mkdir -p /home/tianhangzhu/logs/$experiment_name

log_file="/home/tianhangzhu/logs/$experiment_name/log.txt"

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
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.default_local_dir=/mnt/shared-nfs/home/tianhangzhu/output_copy/$experiment_name \
    trainer.project_name=leadersftv4 \
    trainer.experiment_name=$experiment_name \
    trainer.logger=['console','wandb'] \
    trainer.resume_mode=auto \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true $@ > >(tee -a $log_file) 2>&1 &
