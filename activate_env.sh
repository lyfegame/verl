#!/bin/bash

# Auto-activation script for VERL repository
# This script should be sourced when entering the repository directory

VENV_NAME="verl_fixed"

# Check if we're already in the correct environment
if [[ "$CONDA_DEFAULT_ENV" != "$VENV_NAME" ]]; then
    echo "Switching to VERL environment: $VENV_NAME"
    
    # Check if the environment exists
    if conda info --envs | grep -q "$VENV_NAME"; then
        source ~/anaconda3/etc/profile.d/conda.sh
        conda activate "$VENV_NAME"
        echo "Activated environment: $VENV_NAME"
    else
        echo "Environment $VENV_NAME not found. Please run './setup_env.sh' first."
        return 1
    fi
else
    echo "Already in VERL environment: $VENV_NAME"
fi

# Export PYTHONPATH to include the current directory
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "Ready to run VERL training scripts!"
echo "Current environment: $(conda info --envs | grep '*' | awk '{print $1}')" 