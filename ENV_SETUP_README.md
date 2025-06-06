# VERL Environment Setup Guide

## Quick Start

Since the conda setup was hanging, I've created a faster alternative using Python venv:

### 1. First-time Setup

```bash
# Run the quick setup script (uses Python venv instead of conda)
./quick_setup.sh
```

This will:
- Create a local `.venv` directory in the repository
- Install all required dependencies with compatible versions
- Create an `activate_verl_env.sh` script for manual activation

### 2. Auto-activation Setup

To automatically activate the environment when you enter this repository:

```bash
# Add the auto-activation code to your .bashrc
cat bashrc_addition.txt >> ~/.bashrc

# Reload your shell
source ~/.bashrc
```

Now whenever you `cd` into this repository (or any subdirectory), the VERL environment will automatically activate!

### 3. Manual Activation

If you prefer manual activation:

```bash
# From anywhere:
source /home/ubuntu/sharedusmidwest1/tianhangzhu/codev3/verltianhang/.venv/bin/activate

# Or from within the repository:
source .venv/bin/activate

# Or use the helper script:
./activate_verl_env.sh
```

### 4. Running the Training Script

Once the environment is activated (either automatically or manually), you can run the original script:

```bash
bash examples/grpo_trainer/run_qwen3-8b.sh
```

## Troubleshooting

1. **If setup fails with numpy errors**: The quick_setup.sh script pins numpy to version 1.26.4 which should be compatible.

2. **If you need flash-attn or liger-kernel**: These were skipped for faster setup. Install them manually:
   ```bash
   pip install flash-attn liger-kernel
   ```

3. **To deactivate auto-activation**: Remove the code from your .bashrc or run:
   ```bash
   unset -f cd
   ```

4. **To remove the environment**: Simply delete the .venv directory:
   ```bash
   rm -rf .venv
   ```

## Alternative: Continue with Conda Setup

If you prefer to wait for the conda setup (setup_env.sh), it might take 10-30 minutes but will create a more isolated environment. The auto-activation scripts support both approaches. 