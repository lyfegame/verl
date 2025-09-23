set -x
conda activate verl
export HYDRA_FULL_ERROR=1

# Enhanced debugging and logging
echo "========================================" 
echo "Training started at: $(date)"
echo "========================================" 

# NCCL configuration with verbose debugging
export NCCL_TIMEOUT=1800
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO  # Enable NCCL debug logging
export NCCL_DEBUG_SUBSYS=ALL  # Debug all NCCL subsystems
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # PyTorch distributed debug
export CC=/usr/bin/gcc

# CUDA debugging
export CUDA_LAUNCH_BLOCKING=0  # Set to 1 for synchronous execution (slower but better error messages)
export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"  # H200 compatibility

# Memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8
export TORCH_CUDNN_V8_API_ENABLED=1

# Log environment info
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Monitor system resources before training
echo "========================================" 
echo "System resources BEFORE training:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv
echo "CPU Memory: $(free -h | grep Mem | awk '{print "Total:", $2, "Used:", $3, "Free:", $4}')"
echo "========================================" 

nnodes=1
nproc_per_node=8
lr=4e-4
micro_batch_size_per_gpu=1
train_batch_size=512

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
DISK_ROOT=/home/tianhangzhu/
ROOT_DIR=$DISK_ROOT/codev4/verlrollv2

# Add the verl directory to the Python path
# This ensures that Python can find the verl module when running the script
export PYTHONPATH=$ROOT_DIR/verl:$PYTHONPATH
echo "Added $ROOT_DIR/verl to PYTHONPATH"
########################################################
# Create log directory
initial_model_path=Qwen/Qwen2.5-Coder-0.5B-Instruct

# experiment_name=leadermodel14b_sft22k-lr-1e-4-trainbatch-32-astllmformatidentitymultiturn5xfixedsub10xrest-20250823-024202
experiment_name=leadermodel05b_sft22k-lr-${lr}-trainbatch-${train_batch_size}-firstuserprioritydiverse-allsuccess100xsystem
# train_file=/home/tianhangzhu/gcs_view/home/tianhangzhu/data/sample_train_50.jsonl
# test_file=/home/tianhangzhu/gcs_view/home/tianhangzhu/data/sample_test_50.jsonl
# train_file=/home/tianhangzhu/gcs_view/home/tianhangzhu/data/noninteractive_results_swe_gym_train_rest_v2/ast_llm_output_max_tokens_20000_Noneconvs/leader_train_multiturn_fixedsub5xsub10xother.jsonl
# test_file=/home/tianhangzhu/gcs_view/home/tianhangzhu/data/noninteractive_results_swe_gym_train_rest_v2/ast_llm_output_max_tokens_20000_Noneconvs/leader_test_multiturn_fixedsub5xsub10xother.jsonl
train_file=/home/tianhangzhu/data2/leader_train_multiturn_sys100x_imp35x_impsub35x_firstmsg35x_optimized.jsonl
test_file=/home/tianhangzhu/data2/leader_test_multiturn_sys100x_imp35x_impsub35x_firstmsg35x_optimized.jsonl
# train_file=/home/tianhangzhu/gcs_view/home/tianhangzhu/data/noninteractive_results_swe_gym_train_rest_v2/first_user_priority_output_max_tokens_20000_Noneconvs//leader_train_multiturn_with_upsampling.jsonl
# test_file=/home/tianhangzhu/gcs_view/home/tianhangzhu/data/noninteractive_results_swe_gym_train_rest_v2/first_user_priority_output_max_tokens_20000_Noneconvs//leader_test_multiturn_with_upsampling.jsonl
# train_file=/home/tianhangzhu/gcs_view/home/tianhangzhu/data/noninteractive_results_swe_gym_train_rest_v2/ast_llm_output_max_tokens_20000_Noneconvs/leader_train_multiturn_fixedsub5xsub10xother.jsonl
# test_file=/home/tianhangzhu/gcs_view/home/tianhangzhu/data/noninteractive_results_swe_gym_train_rest_v2/ast_llm_output_max_tokens_20000_Noneconvs/leader_test_multiturn_fixedsub5xsub10xother.jsonl
max_length=22000
# initial_model_path=Qwen/Qwen2.5-14B-Instruct
mkdir -p $DISK_ROOT/logs/$experiment_name

log_file="$DISK_ROOT/logs/$experiment_name/log.txt"

# Write the IP address of the machine at the first line of the log file
hostname -I | awk '{print $1}' > $log_file

# Create a background GPU monitor that logs memory usage every 30 seconds
monitor_gpu() {
    while true; do
        echo "========================================" >> $log_file
        echo "GPU Memory Status at $(date):" >> $log_file
        nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,utilization.memory --format=csv >> $log_file
        echo "CPU Memory: $(free -h | grep Mem | awk '{print "Used:", $3, "Free:", $4}')" >> $log_file
        echo "========================================" >> $log_file
        sleep 30
    done
}

# Start GPU monitor in background
monitor_gpu &
MONITOR_PID=$!
echo "Started GPU monitor with PID: $MONITOR_PID" >> $log_file

# Function to cleanup monitor on exit
cleanup() {
    echo "Cleaning up GPU monitor..." >> $log_file
    kill $MONITOR_PID 2>/dev/null || true
    
    # Final system state
    echo "========================================" >> $log_file
    echo "System resources AFTER training (or failure):" >> $log_file
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv >> $log_file
    echo "CPU Memory: $(free -h | grep Mem | awk '{print "Total:", $2, "Used:", $3, "Free:", $4}')" >> $log_file
    echo "Training ended/failed at: $(date)" >> $log_file
    echo "========================================" >> $log_file
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

echo "Starting torchrun at $(date)" >> $log_file
echo "Configuration:" >> $log_file
echo "  - Model: $initial_model_path" >> $log_file  
echo "  - Sequence Length: $max_length" >> $log_file
echo "  - Batch Size: $train_batch_size" >> $log_file
echo "  - Micro Batch Size: $micro_batch_size_per_gpu" >> $log_file
echo "  - GPUs: $nproc_per_node" >> $log_file
echo "  - Learning Rate: $lr" >> $log_file
echo "  - Train File: $train_file" >> $log_file
echo "========================================" >> $log_file

# Note: model.fsdp_config.model_dtype=bfloat16 is required for Flash Attention 2
# Flash Attention 2 only supports float16 and bfloat16, not float32
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
    model.fsdp_config.model_dtype=bfloat16 \
    trainer.total_epochs=8 \
    model.enable_gradient_checkpointing=True \
    trainer.save_freq=100 \
    trainer.seed=11 \
    trainer.test_freq=100 \
    trainer.default_local_dir=$DISK_ROOT/output_new/$experiment_name \
    trainer.project_name=leadersftv4 \
    trainer.experiment_name=$experiment_name \
    trainer.logger=['console','wandb'] \
    trainer.resume_mode="auto" \
    ++trainer.use_sudo_for_checkpoint=True \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true $@ > >(tee -a $log_file) 2>&1 &
