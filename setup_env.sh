#!/bin/bash

# Setup script for VERL repository
# This creates a new virtual environment with compatible packages

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_NAME="verl_fixed"
VENV_PATH="$HOME/anaconda3/envs/$VENV_NAME"

echo "Setting up VERL environment in $VENV_PATH"

# Remove existing environment if it exists
if conda info --envs | grep -q "$VENV_NAME"; then
    echo "Removing existing $VENV_NAME environment..."
    conda env remove -n "$VENV_NAME" -y
fi

# Create new conda environment with Python 3.10
echo "Creating new conda environment: $VENV_NAME"
conda create -n "$VENV_NAME" python=3.10 -y

# Activate the environment
echo "Activating environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$VENV_NAME"

# Install PyTorch first (this helps avoid conflicts)
echo "Installing PyTorch..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install compatible versions of key packages
echo "Installing compatible packages..."
pip install --no-cache-dir \
    "numpy>=1.24.0,<2.0.0" \
    "transformers>=4.44.0" \
    "accelerate>=0.33.0" \
    "tokenizers>=0.19.0"

# Install flash-attention separately (can be tricky)
echo "Installing flash-attention..."
pip install flash-attn --no-build-isolation

# Install other dependencies
echo "Installing remaining dependencies..."
pip install --no-cache-dir \
    codetiming \
    datasets \
    dill \
    hydra-core \
    liger-kernel \
    pandas \
    peft \
    "pyarrow>=19.0.0" \
    pybind11 \
    pylatexenc \
    pre-commit \
    "ray[default]" \
    "tensordict<=0.6.2" \
    torchdata \
    wandb \
    "packaging>=20.0" \
    uvicorn \
    fastapi

# Install VLLM (specific version that works)
echo "Installing VLLM..."
pip install vllm==0.8.4

# Install the VERL package in development mode
echo "Installing VERL package..."
pip install -e .

echo "Environment setup complete!"
echo "To use this environment, run: conda activate $VENV_NAME" 