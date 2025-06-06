#!/bin/bash

# Auto-activation script for VERL repository
# Add this to your ~/.bashrc or source it manually

# Function to auto-activate VERL environment when entering the repo
auto_activate_verl() {
    local repo_path="/home/ubuntu/sharedusmidwest1/tianhangzhu/codev3/verltianhang"
    local venv_name="verl_fixed"
    
    # Check if we're in the VERL repository directory or any subdirectory
    if [[ "$PWD" == "$repo_path"* ]]; then
        # Check if we're not already in the correct environment
        if [[ "$CONDA_DEFAULT_ENV" != "$venv_name" ]]; then
            # Check if the environment exists
            if conda info --envs 2>/dev/null | grep -q "$venv_name"; then
                echo "🔄 Auto-activating VERL environment: $venv_name"
                source ~/anaconda3/etc/profile.d/conda.sh
                conda activate "$venv_name"
                export PYTHONPATH="$repo_path:$PYTHONPATH"
                echo "✅ Environment activated: $venv_name"
            else
                echo "❌ VERL environment '$venv_name' not found."
                echo "   Run 'cd $repo_path && ./setup_env.sh' to create it."
            fi
        fi
    fi
}

# Hook into the cd command
cd() {
    builtin cd "$@"
    auto_activate_verl
}

# Also check when this script is sourced
auto_activate_verl

echo "🚀 VERL auto-activation enabled!"
echo "   The environment will automatically activate when you enter the repository."
echo "   To disable, restart your shell or run: unset -f cd" 