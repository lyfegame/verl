#!/usr/bin/env python3
"""Test script to verify dataloader functionality."""

import os
import sys
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}", flush=True)

log("Starting dataloader test...")

try:
    # Test imports
    log("Testing imports...")
    import torch
    from transformers import AutoTokenizer
    from omegaconf import OmegaConf
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
    from torch.utils.data import DataLoader
    
    log(f"PyTorch version: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    log(f"GPU count: {torch.cuda.device_count()}")
    
    # Create minimal config
    log("Creating config...")
    config = {
        'cache_dir': '~/.cache/verl/rlhf',
        'prompt_key': 'prompt',
        'max_prompt_length': 512,
        'max_response_length': 512,
        'truncation': 'error',
        'return_raw_chat': False,
        'filter_overlong_prompts': True,
    }
    config = OmegaConf.create(config)
    
    # Load tokenizer
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    log("Creating dataset...")
    train_dataset = RLHFDataset(
        data_files=os.path.expanduser('~/data/gsm8k/train.parquet'),
        tokenizer=tokenizer,
        config=config,
        processor=None,
    )
    
    log(f"Dataset size: {len(train_dataset)}")
    
    # Create dataloader
    log("Creating dataloader...")
    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        num_workers=0,  # Use 0 for debugging
        drop_last=True,
        collate_fn=collate_fn,
        shuffle=True,
    )
    
    log(f"Dataloader batches: {len(dataloader)}")
    
    # Test loading batches
    log("Testing batch loading...")
    for i, batch in enumerate(dataloader):
        log(f"Loaded batch {i+1}/{min(3, len(dataloader))}")
        log(f"  Keys: {list(batch.keys())}")
        log(f"  Batch size: {batch['input_ids'].shape[0]}")
        log(f"  Input shape: {batch['input_ids'].shape}")
        log(f"  Attention mask shape: {batch['attention_mask'].shape}")
        
        if i >= 2:  # Only test first 3 batches
            break
    
    log("Dataloader test completed successfully!")
    
except Exception as e:
    log(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 