# Excel Validation System for VERL

This system enables validation of Excel manipulation tasks using VERL's PPO infrastructure with custom datasets and reward functions.

## Overview

The system consists of:
1. **Custom Excel Dataset** (`verl/utils/dataset/excel_dataset.py`) - Loads Excel files and extracts content for model input
2. **Custom Excel Reward Function** (`verl/utils/reward/excel_reward.py`) - Compares generated Excel files with reference files
3. **Modified main_ppo.py** - Adds validation-only mode to the existing PPO trainer
4. **Helper Scripts** - For easy execution

## Components

### 1. Excel Dataset (ExcelTaskDataset)
- Loads tasks from JSONL files containing Excel manipulation instructions
- Extracts Excel content (data, formulas, tables, charts) similar to the JavaScript UI
- Formats data as CSV with formulas shown in parentheses
- Stores metadata in non_tensor_batch for reward computation

### 2. Excel Reward Function
- Compares final Excel files with reference files
- Supports multiple verification criteria:
  - Content comparison
  - Conditional formatting
  - Tables
  - Frozen panes
- Returns partial rewards based on the types of differences found
- Expects metadata in DataProto's non_tensor_batch field

### 3. Validation-Only Mode
The modified `main_ppo.py` now supports a `validation_only` flag:
- When `validation_only=true`, it runs validation and exits without training
- Useful for evaluating pre-trained models on Excel tasks

## Setup

### 1. Install Required Dependencies

```bash
# Ensure you have openpyxl for Excel file handling
pip install openpyxl pandas
```

### 2. Create Test Excel Files (Optional)

```bash
# Run the test file creation script
python create_test_excel.py
```

### 3. Run Integration Tests

```bash
# Test the integration
python test_excel_integration.py
```

## Usage

### Method 1: Using main_ppo.py with validation_only flag

```bash
# For validation only
python3 -m verl.trainer.main_ppo \
    validation_only=true \
    data.use_excel_dataset=true \
    data.train_files=/path/to/excel_tasks.jsonl \
    data.val_files=/path/to/excel_tasks.jsonl \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    custom_reward_function.path=verl/utils/reward/excel_reward.py \
    custom_reward_function.name=excel_comparison_reward \
    reward_model.enable=false \
    # ... other configs
```

### Method 2: Using the convenience script

```bash
# Make script executable
chmod +x run_excel_validation_ppo.sh

# Run validation
./run_excel_validation_ppo.sh
```

### Method 3: For full training on Excel tasks

Simply set `validation_only=false` or omit it:

```bash
python3 -m verl.trainer.main_ppo \
    data.use_excel_dataset=true \
    data.train_files=/path/to/excel_tasks.jsonl \
    # ... other configs (validation_only defaults to false)
```

## Dataset Format

The JSONL dataset should have entries like:

```json
{
  "task_id": "task_36",
  "user_query": "In the Payments worksheet, add a header to column F...",
  "initial_file": "/path/to/initial.xlsx",
  "final_file": "/path/to/final.xlsx",
  "reference_file": "/path/to/reference.xlsx",
  "verifiers_dict": {
    "check_content": true,
    "check_conditional_formatting": true,
    "check_tables": true,
    "check_frozen_panes": true
  }
}
```

## Key Configuration Options

- `validation_only`: Set to `true` for validation-only mode
- `data.use_excel_dataset`: Set to `true` to use the Excel dataset
- `custom_reward_function.path`: Path to the reward function file
- `custom_reward_function.name`: Name of the reward function
- `reward_model.enable`: Set to `false` when using custom reward function

## Data Flow

1. **Dataset Loading**: ExcelTaskDataset loads JSONL files and Excel content
2. **Batch Creation**: Data is formatted with tensors and non-tensor metadata
3. **Collation**: VERL's default collate_fn handles batching
4. **DataProto**: Batches are converted to DataProto with metadata in non_tensor_batch
5. **Generation**: Model generates responses
6. **Reward Computation**: Excel reward function accesses metadata from non_tensor_batch

## Debugging Tips

### 1. Check Dataset Loading
```python
from verl.utils.dataset.excel_dataset import ExcelTaskDataset
dataset = ExcelTaskDataset([data_file], tokenizer, processor, config)
item = dataset[0]
print(item.keys())  # Should include excel_metadata
```

### 2. Verify DataProto Structure
```python
# After collation
data_proto = DataProto.from_single_dict(collated_batch)
print(data_proto.non_tensor_batch.keys())  # Should include excel_metadata
```

### 3. Test Reward Function
```python
from verl.utils.reward.excel_reward import excel_comparison_reward
result = excel_comparison_reward(data_proto)
print(result['reward_tensor'].shape)  # Should be (batch_size, 1)
```

## Common Issues and Solutions

1. **"No excel_metadata found in non_tensor_batch"**
   - Ensure dataset returns excel_metadata as numpy array with dtype=object
   - Check that collate_fn preserves non-tensor data

2. **Shape mismatch in rewards**
   - Reward tensor should be shape (batch_size, 1) not (batch_size,)
   - The reward function includes unsqueeze(-1) to fix this

3. **File not found errors**
   - Ensure all Excel files in the dataset exist
   - The reward function currently expects pre-generated final_file
   - In production, you'd generate this from model output

## Integration with Existing VERL Workflow

This system seamlessly integrates with VERL's existing infrastructure:
- Uses the same command-line interface as regular PPO training
- Compatible with all existing model configurations
- Supports distributed training when `validation_only=false`
- Can switch between Excel and regular datasets using `data.use_excel_dataset`

## Example: Validating Qwen3-8B on Excel Tasks

```bash
python3 -m verl.trainer.main_ppo \
    validation_only=true \
    algorithm.adv_estimator=grpo \
    data.use_excel_dataset=true \
    data.train_files=/home/ubuntu/data/excel_tasks.jsonl \
    data.val_files=/home/ubuntu/data/excel_tasks.jsonl \
    data.train_batch_size=32 \
    data.max_prompt_len=4096 \
    data.max_response_len=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    custom_reward_function.path=verl/utils/reward/excel_reward.py \
    custom_reward_function.name=excel_comparison_reward \
    reward_model.enable=false \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1
```

This will load the Excel tasks, run the model to generate responses, and evaluate them using the Excel comparison reward function. 