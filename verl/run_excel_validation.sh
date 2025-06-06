#!/bin/bash

# Script to run Excel validation with VERL
# This demonstrates how to use the custom Excel dataset and reward function

echo "Excel Validation Runner for VERL"
echo "================================"

# Default values
MODE="validate"  # "validate" or "train"
CONFIG_PATH="trainer/config/excel_ppo_trainer.yaml"
MODEL_PATH=""
DATA_PATH="/home/ubuntu/sharedusmidwest1/tianhangzhu/codev3/prototype-new-infra_ourtv1/evaluation/tasks/rm_data/merged_test83_72_data_noread_addr.jsonl"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --data)
            DATA_PATH="$2"
            shift 2
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode MODE      Run mode: 'validate' (default) or 'train'"
            echo "  --model PATH     Path to the model checkpoint"
            echo "  --data PATH      Path to the JSONL dataset file"
            echo "  --config PATH    Path to config file (default: trainer/config/excel_ppo_trainer.yaml)"
            echo "  --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Run validation only"
            echo "  $0 --mode validate --model meta-llama/Llama-3-8b-hf"
            echo ""
            echo "  # Run full training"
            echo "  $0 --mode train --model meta-llama/Llama-3-8b-hf --data /path/to/data.jsonl"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if model path is provided
if [ -z "$MODEL_PATH" ]; then
    echo "Error: Model path is required. Use --model to specify."
    exit 1
fi

# Create a temporary config override file
OVERRIDE_FILE=$(mktemp /tmp/excel_config_override.XXXXXX.yaml)

# Build override configuration
cat > "$OVERRIDE_FILE" << EOF
# Auto-generated config overrides
validation_only: $([ "$MODE" = "validate" ] && echo "true" || echo "false")

actor_rollout_ref:
  model:
    path: $MODEL_PATH

critic:
  model:
    path: $MODEL_PATH

data:
  train_files:
    - $DATA_PATH
  val_files:
    - $DATA_PATH
EOF

echo "Running in $MODE mode..."
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Config: $CONFIG_PATH"
echo ""

# Run the validation/training
if [ "$MODE" = "validate" ]; then
    echo "Starting validation..."
    python trainer/main_validate.py \
        --config-path="$(dirname $CONFIG_PATH)" \
        --config-name="$(basename $CONFIG_PATH .yaml)" \
        --config-dir="$(pwd)/$(dirname $CONFIG_PATH)" \
        hydra.run.dir="./outputs/excel_validation_$(date +%Y%m%d_%H%M%S)" \
        +hydra/job_logging=colorlog \
        hydra/hydra_logging=colorlog \
        $(cat "$OVERRIDE_FILE" | sed 's/^/++/g' | tr '\n' ' ')
else
    echo "Starting training..."
    python trainer/main_validate.py \
        --config-path="$(dirname $CONFIG_PATH)" \
        --config-name="$(basename $CONFIG_PATH .yaml)" \
        --config-dir="$(pwd)/$(dirname $CONFIG_PATH)" \
        hydra.run.dir="./outputs/excel_training_$(date +%Y%m%d_%H%M%S)" \
        trainer.num_epochs=10 \
        trainer.save_interval=100 \
        wandb.enabled=true \
        +hydra/job_logging=colorlog \
        hydra/hydra_logging=colorlog \
        $(cat "$OVERRIDE_FILE" | sed 's/^/++/g' | tr '\n' ' ')
fi

# Clean up
rm -f "$OVERRIDE_FILE"

echo ""
echo "Done!" 