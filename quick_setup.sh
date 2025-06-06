#!/bin/bash

# Quick setup script for VERL repository using Python venv
# This is faster than conda but still creates an isolated environment

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_NAME="verl_env"
VENV_PATH="$REPO_DIR/.venv"

echo "Setting up VERL environment in $VENV_PATH"

# Remove existing environment if it exists
if [ -d "$VENV_PATH" ]; then
    echo "Removing existing environment..."
    rm -rf "$VENV_PATH"
fi

# Create new virtual environment with system Python
echo "Creating new virtual environment..."
python3 -m venv "$VENV_PATH"

# Activate the environment
echo "Activating environment..."
source "$VENV_PATH/bin/activate"

# Upgrade pip first
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version for faster installation, change if you need GPU)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install critical packages with specific versions to avoid numpy issues
echo "Installing core packages..."
pip install --no-cache-dir \
    "numpy==1.26.4" \
    "transformers==4.44.2" \
    "accelerate==0.33.0" \
    "tokenizers==0.19.1"

# Install remaining dependencies
echo "Installing other dependencies..."
pip install --no-cache-dir \
    codetiming \
    datasets \
    dill \
    hydra-core \
    pandas \
    peft \
    "pyarrow>=19.0.0" \
    pybind11 \
    pylatexenc \
    "ray[default]" \
    "tensordict<=0.6.2" \
    torchdata \
    wandb \
    "packaging>=20.0" \
    uvicorn \
    fastapi

# Skip flash-attn and liger-kernel for now (they can be slow to install)
echo "Note: Skipping flash-attn and liger-kernel for quick setup."
echo "You can install them later with: pip install flash-attn liger-kernel"

# Install VLLM
echo "Installing VLLM..."
pip install vllm==0.6.4.post1

# Install the VERL package in development mode
echo "Installing VERL package..."
pip install -e .

echo "Environment setup complete!"
echo "To activate this environment, run: source $VENV_PATH/bin/activate"

# Create activation script
cat > activate_verl_env.sh << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
echo "VERL environment activated!"
EOF
chmod +x activate_verl_env.sh

echo "Created activation script: ./activate_verl_env.sh" 