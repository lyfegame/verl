"""
Simple streaming dataset that doesn't build an index upfront.
Just seeks to the approximate position and reads from there.
"""

import json
import os
from typing import List, Union, Optional, Dict, Any
from torch.utils.data import Dataset, IterableDataset
import torch


class SimpleStreamingDataset(IterableDataset):
    """
    A truly streaming dataset that doesn't load anything upfront.
    This uses IterableDataset which is better suited for streaming large files.
    """
    
    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer=None,  # Not used for pre-tokenized data
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the streaming dataset.
        
        Args:
            parquet_files: Path(s) to JSONL files
            tokenizer: Not used for pre-tokenized data
            config: Configuration dictionary
        """
        config = config or {}
        self.max_length = config.get("max_length", 22000)
        self.truncation = config.get("truncation", "right")
        
        if not isinstance(parquet_files, list):
            parquet_files = [parquet_files]
        
        self.files = parquet_files
        print(f"SimpleStreamingDataset initialized with {len(self.files)} file(s)")
        
    def __iter__(self):
        """Iterate through all files and yield samples."""
        for file_path in self.files:
            print(f"Streaming from: {file_path}")
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f):
                    if line_num % 1000 == 0:
                        print(f"  Processed {line_num} samples...", end='\r')
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Handle pre-tokenized data
                        if 'input_ids' in data and 'loss_mask' in data:
                            input_ids = torch.tensor(data['input_ids'], dtype=torch.long)
                            loss_mask = torch.tensor(data['loss_mask'], dtype=torch.bool)
                            
                            # Apply truncation if needed
                            if len(input_ids) > self.max_length:
                                if self.truncation == 'error':
                                    print(f"Warning: Sequence length {len(input_ids)} exceeds max_length {self.max_length}, skipping")
                                    continue
                                elif self.truncation == 'left':
                                    input_ids = input_ids[-self.max_length:]
                                    loss_mask = loss_mask[-self.max_length:]
                                else:  # right
                                    input_ids = input_ids[:self.max_length]
                                    loss_mask = loss_mask[:self.max_length]
                            
                            yield {
                                'input_ids': input_ids,
                                'loss_mask': loss_mask
                            }
                        else:
                            # Skip non-tokenized data for now
                            continue
                            
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON at line {line_num}: {e}")
                        continue


class IndexedStreamingDataset(Dataset):
    """
    A compromise between full streaming and full loading.
    Only counts lines upfront (much faster than building full index).
    Then seeks to approximate positions when needed.
    """
    
    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer=None,
        config: Optional[Dict[str, Any]] = None
    ):
        config = config or {}
        self.max_length = config.get("max_length", 22000)
        self.truncation = config.get("truncation", "right")
        
        if not isinstance(parquet_files, list):
            parquet_files = [parquet_files]
        
        self.files = parquet_files
        self.file_handles = []
        self.file_lengths = []
        self.cumulative_lengths = [0]
        
        print(f"IndexedStreamingDataset: Counting lines in {len(self.files)} file(s)...")
        
        # Just count lines - much faster than building full index
        for file_path in self.files:
            print(f"  Counting lines in {file_path}...")
            line_count = 0
            with open(file_path, 'r') as f:
                for _ in f:
                    line_count += 1
            
            self.file_lengths.append(line_count)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + line_count)
            print(f"    Found {line_count:,} lines")
        
        self.total_length = self.cumulative_lengths[-1]
        print(f"Total dataset size: {self.total_length:,} samples")
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        """Get a specific sample by index."""
        if idx >= self.total_length:
            raise IndexError(f"Index {idx} out of range")
        
        # Find which file
        file_idx = 0
        for i in range(len(self.cumulative_lengths) - 1):
            if self.cumulative_lengths[i] <= idx < self.cumulative_lengths[i + 1]:
                file_idx = i
                break
        
        # Calculate line number within file
        line_idx = idx - self.cumulative_lengths[file_idx]
        
        # Read the specific line
        with open(self.files[file_idx], 'r') as f:
            for i, line in enumerate(f):
                if i == line_idx:
                    line = line.strip()
                    if not line:
                        return self.__getitem__((idx + 1) % self.total_length)
                    
                    try:
                        data = json.loads(line)
                        
                        if 'input_ids' in data and 'loss_mask' in data:
                            input_ids = torch.tensor(data['input_ids'], dtype=torch.long)
                            loss_mask = torch.tensor(data['loss_mask'], dtype=torch.bool)
                            
                            # Apply truncation
                            if len(input_ids) > self.max_length:
                                if self.truncation == 'error':
                                    raise ValueError(f"Sequence too long: {len(input_ids)}")
                                elif self.truncation == 'left':
                                    input_ids = input_ids[-self.max_length:]
                                    loss_mask = loss_mask[-self.max_length:]
                                else:
                                    input_ids = input_ids[:self.max_length]
                                    loss_mask = loss_mask[:self.max_length]
                            
                            return {
                                'input_ids': input_ids,
                                'loss_mask': loss_mask
                            }
                        else:
                            # Return empty sample if not pre-tokenized
                            return {
                                'input_ids': torch.zeros(1, dtype=torch.long),
                                'loss_mask': torch.zeros(1, dtype=torch.bool)
                            }
                    
                    except json.JSONDecodeError:
                        return self.__getitem__((idx + 1) % self.total_length)
        
        # Should not reach here
        raise RuntimeError(f"Could not find line {line_idx} in file {file_idx}")


# Use the indexed version as the default since FSDP trainer expects __len__
StreamingSFTDataset = IndexedStreamingDataset
