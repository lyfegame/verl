"""
Streaming SFT Dataset for handling large JSONL files efficiently.
This dataset loads data on-demand rather than loading everything into memory upfront.
"""

import json
import os
from typing import List, Union, Optional, Dict, Any
from torch.utils.data import Dataset
import torch


class StreamingSFTDataset(Dataset):
    """
    A memory-efficient dataset that streams JSONL files line by line.
    Only counts lines upfront (fast), then reads specific lines on-demand.
    """
    
    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer=None,  # Not used for pre-tokenized data
        config: Optional[Dict[str, Any]] = None
    ):
        config = config or {}
        self.max_length = config.get("max_length", 22000)
        self.truncation = config.get("truncation", "right")
        assert self.truncation in ["error", "left", "right"]
        
        if not isinstance(parquet_files, list):
            parquet_files = [parquet_files]
        
        self.files = parquet_files
        self.file_lengths = []
        self.cumulative_lengths = [0]
        
        # Just count lines - MUCH faster than building full index
        print(f"StreamingSFTDataset: Counting lines in {len(self.files)} file(s)...")
        for file_path in self.files:
            print(f"  Counting: {file_path}")
            line_count = sum(1 for _ in open(file_path, 'r'))
            self.file_lengths.append(line_count)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + line_count)
            file_size = os.path.getsize(file_path) / (1024**3)
            print(f"    {line_count:,} lines, {file_size:.2f} GB")
        
        self.total_length = self.cumulative_lengths[-1]
        print(f"Total dataset size: {self.total_length:,} samples")
    
    def __len__(self):
        return self.total_length
    
    def _find_file_and_line(self, idx):
        """Find which file and line number corresponds to the global index."""
        file_idx = 0
        for i in range(len(self.cumulative_lengths) - 1):
            if self.cumulative_lengths[i] <= idx < self.cumulative_lengths[i + 1]:
                file_idx = i
                break
        
        line_idx = idx - self.cumulative_lengths[file_idx]
        return file_idx, line_idx
    
    def __getitem__(self, idx):
        """Get a specific sample by index - just read the nth line."""
        if idx >= self.total_length:
            raise IndexError(f"Index {idx} out of range")
        
        # Find which file and line
        file_idx, line_idx = self._find_file_and_line(idx)
        
        # Read the specific line directly
        with open(self.files[file_idx], 'r') as f:
            for i, line in enumerate(f):
                if i == line_idx:
                    line = line.strip()
                    if not line:
                        # Empty line, return next
                        return self.__getitem__((idx + 1) % self.total_length)
                    
                    try:
                        data = json.loads(line)
                        
                        # Handle pre-tokenized data
                        if 'input_ids' in data and 'loss_mask' in data:
                            input_ids = torch.tensor(data['input_ids'], dtype=torch.long)
                            loss_mask = torch.tensor(data['loss_mask'], dtype=torch.bool)
                            
                            # Apply truncation if needed
                            if len(input_ids) > self.max_length:
                                if self.truncation == 'error':
                                    raise ValueError(f"Sequence length {len(input_ids)} exceeds max_length {self.max_length}")
                                elif self.truncation == 'left':
                                    input_ids = input_ids[-self.max_length:]
                                    loss_mask = loss_mask[-self.max_length:]
                                else:  # right
                                    input_ids = input_ids[:self.max_length]
                                    loss_mask = loss_mask[:self.max_length]
                            
                            # Pad to max_length for batching
                            if len(input_ids) < self.max_length:
                                pad_length = self.max_length - len(input_ids)
                                input_ids = torch.cat([
                                    input_ids,
                                    torch.zeros(pad_length, dtype=torch.long)
                                ])
                                loss_mask = torch.cat([
                                    loss_mask,
                                    torch.zeros(pad_length, dtype=torch.bool)
                                ])
                            
                            return {
                                'input_ids': input_ids,
                                'loss_mask': loss_mask
                            }
                        else:
                            # Not pre-tokenized, skip
                            print(f"Warning: Sample {idx} not pre-tokenized, skipping")
                            return self.__getitem__((idx + 1) % self.total_length)
                    
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON at index {idx}: {e}")
                        return self.__getitem__((idx + 1) % self.total_length)
        
        # Should not reach here
        raise RuntimeError(f"Could not find line {line_idx} in file {file_idx}")


class StreamingMultiTurnSFTDataset(StreamingSFTDataset):
    """
    Placeholder for compatibility - just uses the base streaming dataset.
    """
    pass