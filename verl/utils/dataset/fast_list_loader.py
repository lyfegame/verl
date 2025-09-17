#!/usr/bin/env python3
"""
Fast List Loader Dataset - Loads all data into memory using parallel processing
for fastest possible __getitem__ access.
"""

from typing import List, Union
import json
import os
import time
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer


def load_file_chunk_to_list(args):
    """Load a chunk of lines from a file into a list (for parallel processing)."""
    file_path, start_line, end_line, chunk_id = args
    
    data_chunk = []
    with open(file_path, 'r', buffering=8192*8) as f:  # Large buffer for better I/O
        for i, line in enumerate(f):
            if i < start_line:
                continue
            if end_line is not None and i >= end_line:
                break
            
            line = line.strip()
            if line:
                try:
                    data_chunk.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip malformed JSON lines
                    continue
    
    return chunk_id, data_chunk


def count_lines_fast(file_path):
    """Count lines efficiently using binary reading."""
    line_count = 0
    with open(file_path, 'rb', buffering=8192*8) as f:
        buffer = f.read(8192*8)
        while buffer:
            line_count += buffer.count(b'\n')
            buffer = f.read(8192*8)
    return line_count


class FastListDataset(Dataset):
    """
    Dataset that loads all data into memory using parallel processing.
    Optimized for fastest possible __getitem__ access at the cost of memory usage.
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer,
                 prompt_key='prompt',
                 prompt_dict_keys=None,
                 response_key='response', 
                 response_dict_keys=None,
                 max_length=1024,
                 truncation='error',
                 num_workers=None):
        """
        Initialize with parallel data loading into memory.
        
        Args:
            parquet_files: JSONL file paths
            tokenizer: Tokenizer instance or path
            max_length: Maximum sequence length
            truncation: How to handle sequences > max_length ('error', 'left', 'right')
            num_workers: Number of workers for parallel loading (None = auto)
        """
        assert truncation in ['error', 'left', 'right']
        self.truncation = truncation
        self.max_length = max_length

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        
        # Auto-detect number of workers
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 8)
        self.num_workers = max(1, num_workers)
        
        # For compatibility
        self.prompt_key = prompt_key
        self.response_key = response_key
        
        # Load all data into memory using parallel processing
        self._load_all_data_parallel()

    def _load_all_data_parallel(self):
        """Load all data into memory using parallel processing."""
        print(f"FastListDataset: Loading {len(self.parquet_files)} file(s) using {self.num_workers} workers...")
        start_time = time.time()
        
        all_data = []
        
        for file_path in self.parquet_files:
            file_size_mb = os.path.getsize(file_path) / (1024*1024)
            print(f"  Loading {os.path.basename(file_path)} ({file_size_mb:.1f} MB)...")
            
            # Count lines efficiently
            total_lines = count_lines_fast(file_path)
            print(f"    Found {total_lines:,} lines")
            
            if total_lines == 0:
                continue
            
            # Calculate line ranges for each worker
            lines_per_worker = max(1, total_lines // self.num_workers)
            
            # Create chunks for parallel processing
            chunks = []
            for i in range(self.num_workers):
                start_line = i * lines_per_worker
                if i == self.num_workers - 1:  # Last worker takes remaining lines
                    end_line = total_lines
                else:
                    end_line = (i + 1) * lines_per_worker
                
                if start_line < total_lines:
                    chunks.append((file_path, start_line, end_line, i))
            
            # Process chunks in parallel
            chunk_results = [[] for _ in range(len(chunks))]
            
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(load_file_chunk_to_list, chunk): idx 
                          for idx, chunk in enumerate(chunks)}
                
                for future in as_completed(futures):
                    chunk_idx = futures[future]
                    chunk_id, chunk_data = future.result()
                    chunk_results[chunk_id] = chunk_data
            
            # Combine results in order to maintain data order
            file_data = []
            for chunk_data in chunk_results:
                file_data.extend(chunk_data)
            
            all_data.extend(file_data)
            print(f"    Loaded {len(file_data):,} items from {os.path.basename(file_path)}")
        
        elapsed = time.time() - start_time
        print(f"Data loading completed in {elapsed:.2f}s ({len(all_data):,} total items)")
        
        # Store data and preprocess
        self.data = all_data
        self._preprocess_data()

    def _preprocess_data(self):
        """Preprocess all data for faster access."""
        print("Preprocessing data...")
        start_time = time.time()
        
        self.input_ids = []
        self.loss_masks = []
        
        for item in self.data:
            if 'input_ids' in item and 'loss_mask' in item:
                self.input_ids.append(item['input_ids'])
                self.loss_masks.append(item['loss_mask'])
            else:
                # Handle missing data with dummy values
                self.input_ids.append([self.tokenizer.pad_token_id])
                self.loss_masks.append([0])
        
        elapsed = time.time() - start_time
        print(f"Preprocessing completed in {elapsed:.2f}s")

    def __len__(self):
        """Return total number of samples."""
        return len(self.data)

    def __getitem__(self, item):
        """Get sample by index - optimized for speed with pre-loaded data."""
        # Convert to tensors (this is the main processing cost)
        input_ids = torch.tensor(self.input_ids[item], dtype=torch.long)
        loss_mask = torch.tensor(self.loss_masks[item], dtype=torch.long)
        
        sequence_length = input_ids.shape[0]
        
        # Handle truncation
        if sequence_length > self.max_length:
            if self.truncation == 'error':
                raise ValueError(f'{sequence_length=} is larger than {self.max_length=}')
            elif self.truncation == 'left':
                input_ids = input_ids[-self.max_length:]
                loss_mask = loss_mask[-self.max_length:]
            else:  # 'right'
                input_ids = input_ids[:self.max_length]
                loss_mask = loss_mask[:self.max_length]
            
            sequence_length = self.max_length
        
        # Create attention mask
        attention_mask = torch.ones(sequence_length, dtype=torch.long)
        
        # Apply padding if needed
        if sequence_length < self.max_length:
            pad_length = self.max_length - sequence_length
            
            padded_input_ids = torch.ones(pad_length, dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            input_ids = torch.cat((input_ids, padded_input_ids))
            
            padded_attention_mask = torch.zeros(pad_length, dtype=attention_mask.dtype)
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            
            padded_loss_mask = torch.zeros(pad_length, dtype=loss_mask.dtype)
            loss_mask = torch.cat((loss_mask, padded_loss_mask))
        
        # Compute position IDs
        position_ids = compute_position_id_with_mask(attention_mask)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask
        }


# Alternative version with even more aggressive optimization
class UltraFastListDataset(Dataset):
    """
    Ultra-optimized version that pre-computes ALL tensors during initialization.
    Trades initialization time and memory for fastest possible __getitem__.
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer,
                 prompt_key='prompt',
                 prompt_dict_keys=None,
                 response_key='response',
                 response_dict_keys=None,
                 max_length=1024,
                 truncation='error',
                 num_workers=None):
        
        assert truncation in ['error', 'left', 'right']
        self.truncation = truncation
        self.max_length = max_length

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 8)
        self.num_workers = max(1, num_workers)
        
        # Load and preprocess ALL data
        self._load_and_preprocess_all()

    def _load_and_preprocess_all(self):
        """Load all data and pre-compute all tensors."""
        print(f"UltraFastListDataset: Pre-computing ALL tensors using {self.num_workers} workers...")
        start_time = time.time()
        
        # First, load raw data using parallel processing (reuse FastListDataset logic)
        temp_dataset = FastListDataset(
            parquet_files=self.parquet_files,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            truncation=self.truncation,
            num_workers=self.num_workers
        )
        
        print("Pre-computing all tensors...")
        preprocess_start = time.time()
        
        # Pre-compute all tensors
        self.precomputed_samples = []
        
        for i in range(len(temp_dataset.data)):
            if i % 1000 == 0:
                print(f"  Progress: {i:,}/{len(temp_dataset.data):,} ({100*i/len(temp_dataset.data):.1f}%)", end='\r')
            
            # Use the FastListDataset's __getitem__ to get processed sample
            sample = temp_dataset[i]
            self.precomputed_samples.append(sample)
        
        preprocess_time = time.time() - preprocess_start
        total_time = time.time() - start_time
        
        print(f"\nTensor pre-computation completed in {preprocess_time:.2f}s")
        print(f"Total initialization time: {total_time:.2f}s")
        print(f"Memory usage: ~{len(self.precomputed_samples) * self.max_length * 4 * 4 / (1024*1024*1024):.1f} GB")

    def __len__(self):
        return len(self.precomputed_samples)

    def __getitem__(self, item):
        """Ultra-fast access - just return pre-computed tensor."""
        return self.precomputed_samples[item]


if __name__ == "__main__":
    """Test the fast list datasets"""
    
    TEST_FILE = "/home/tianhangzhu/gcs_view/home/tianhangzhu/data/all_success_100_80_v2_processing/all_success_100_80_v2_compacted_data/leader_test_multiturn_optimized.jsonl"
    
    class MockTokenizer:
        pad_token_id = 0
        eos_token_id = 1
    
    print("Testing FastListDataset vs UltraFastListDataset")
    print("="*80)
    
    # Test FastListDataset
    print("\n1. Testing FastListDataset:")
    start = time.time()
    dataset1 = FastListDataset(
        parquet_files=[TEST_FILE],
        tokenizer=MockTokenizer(),
        max_length=25000,
        truncation='right',
        num_workers=4
    )
    load_time1 = time.time() - start
    print(f"FastListDataset loaded in {load_time1:.2f}s, size: {len(dataset1):,}")
    
    # Test access speed
    start = time.time()
    for i in range(min(100, len(dataset1))):
        sample = dataset1[i]
    access_time1 = (time.time() - start) / min(100, len(dataset1)) * 1000
    print(f"Average access time: {access_time1:.2f}ms per item")
    
    # Test UltraFastListDataset
    print(f"\n2. Testing UltraFastListDataset:")
    start = time.time()
    dataset2 = UltraFastListDataset(
        parquet_files=[TEST_FILE],
        tokenizer=MockTokenizer(),
        max_length=25000,
        truncation='right',
        num_workers=4
    )
    load_time2 = time.time() - start
    print(f"UltraFastListDataset loaded in {load_time2:.2f}s, size: {len(dataset2):,}")
    
    # Test access speed
    start = time.time()
    for i in range(min(100, len(dataset2))):
        sample = dataset2[i]
    access_time2 = (time.time() - start) / min(100, len(dataset2)) * 1000
    print(f"Average access time: {access_time2:.2f}ms per item")
    
    print(f"\nComparison:")
    print(f"FastListDataset:      Load={load_time1:.2f}s, Access={access_time1:.2f}ms")
    print(f"UltraFastListDataset: Load={load_time2:.2f}s, Access={access_time2:.2f}ms")
    print(f"Access speedup: {access_time1/access_time2:.1f}x faster")

