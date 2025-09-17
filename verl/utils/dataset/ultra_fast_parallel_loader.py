#!/usr/bin/env python3
"""
Ultra Fast Parallel Loader - Based on streaming_sft_dataset_v2's efficient byte-based chunking
but loads everything into memory for fastest possible access.
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


def load_file_chunk_bytes(args):
    """
    Load a chunk of a file using byte-based seeking (like streaming_sft_dataset_v2).
    This is much more efficient than line-based chunking.
    """
    file_path, start_byte, end_byte, chunk_id = args
    
    data_chunk = []
    with open(file_path, 'rb') as f:
        # Seek DIRECTLY to start byte (no sequential reading!)
        f.seek(start_byte)
        
        # If not starting at beginning, skip to next newline
        if start_byte > 0:
            f.readline()
        
        # Now read only our assigned chunk
        while f.tell() < end_byte:
            line = f.readline()
            if not line:
                break
            
            line_str = line.decode('utf-8', errors='ignore').strip()
            if line_str:
                try:
                    data_chunk.append(json.loads(line_str))
                except json.JSONDecodeError:
                    # Skip malformed JSON lines
                    continue
                    
    return chunk_id, data_chunk


def load_file_simple_optimized(file_path):
    """
    Simple optimized file loading with larger buffer for comparison.
    """
    data = []
    with open(file_path, 'r', buffering=8192*16) as f:  # 128KB buffer
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


class UltraFastParallelLoader(Dataset):
    """
    Ultra-fast parallel loader using byte-based chunking from streaming_sft_dataset_v2.
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
        
        # Auto-detect number of workers
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 8)
        self.num_workers = max(1, num_workers)
        
        # For compatibility
        self.prompt_key = prompt_key
        self.response_key = response_key
        
        # Load all data using optimized byte-based parallel processing
        self._load_all_data_byte_parallel()

    def _load_all_data_byte_parallel(self):
        """Load all data using byte-based parallel processing like streaming_sft_dataset_v2."""
        print(f"UltraFastParallelLoader: Loading {len(self.parquet_files)} file(s) using {self.num_workers} workers (byte-based)...")
        start_time = time.time()
        
        all_data = []
        
        for file_path in self.parquet_files:
            file_size = os.path.getsize(file_path)
            file_size_mb = file_size / (1024*1024)
            print(f"  Loading {os.path.basename(file_path)} ({file_size_mb:.1f} MB)...")
            
            if file_size < 50 * 1024 * 1024:  # Less than 50MB, don't parallelize
                print(f"    Small file, using sequential loading...")
                file_data = load_file_simple_optimized(file_path)
            else:
                print(f"    Large file, using byte-based parallel loading...")
                file_data = self._load_single_file_byte_parallel(file_path)
            
            all_data.extend(file_data)
            print(f"    Loaded {len(file_data):,} items from {os.path.basename(file_path)}")
        
        elapsed = time.time() - start_time
        print(f"Data loading completed in {elapsed:.2f}s ({len(all_data):,} total items)")
        
        # Store data and preprocess
        self.data = all_data
        self._preprocess_data()

    def _load_single_file_byte_parallel(self, file_path):
        """Load a single file using byte-based parallel processing."""
        file_size = os.path.getsize(file_path)
        
        # Split file into chunks for parallel processing (like streaming_sft_dataset_v2)
        chunk_size = max(file_size // self.num_workers, 1024*1024)  # At least 1MB per chunk
        chunks = []
        for i in range(self.num_workers):
            start = i * chunk_size
            end = file_size if i == self.num_workers - 1 else (i + 1) * chunk_size
            if start < file_size:  # Only add valid chunks
                chunks.append((file_path, start, end, i))
        
        print(f"      Split into {len(chunks)} chunks of ~{chunk_size/(1024*1024):.1f}MB each")
        
        # Process chunks in parallel
        all_chunk_data = [[] for _ in range(len(chunks))]
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(load_file_chunk_bytes, chunk): idx 
                      for idx, chunk in enumerate(chunks)}
            
            for future in as_completed(futures):
                chunk_idx = futures[future]
                chunk_id, chunk_data = future.result()
                all_chunk_data[chunk_id] = chunk_data
                print(f"        Chunk {chunk_id}: {len(chunk_data):,} items")
        
        # Combine results in order to maintain data order
        combined_data = []
        for chunk_data in all_chunk_data:
            combined_data.extend(chunk_data)
        
        return combined_data

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
        # Convert to tensors
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


if __name__ == "__main__":
    """Test the ultra-fast parallel loader"""
    
    # Test files
    SMALL_FILE = "/home/tianhangzhu/gcs_view/home/tianhangzhu/data/all_success_100_80_v2_processing/all_success_100_80_v2_compacted_data/leader_test_multiturn_optimized.jsonl"
    LARGE_FILE = "/home/tianhangzhu/gcs_view/home/tianhangzhu/data/all_success_100_80_v2_processing/all_success_100_80_v2_compacted_data/leader_test_multiturn_sys50x_imp35x_impsub35x_firstmsg35x_optimized.jsonl"
    
    class MockTokenizer:
        pad_token_id = 0
        eos_token_id = 1
    
    print("Testing UltraFastParallelLoader vs Simple Loading")
    print("="*80)
    
    for test_name, test_file in [("Small File", SMALL_FILE), ("Large File", LARGE_FILE)]:
        if not os.path.exists(test_file):
            print(f"Skipping {test_name} - file not found")
            continue
            
        file_size_mb = os.path.getsize(test_file) / (1024*1024)
        print(f"\n{test_name}: {os.path.basename(test_file)} ({file_size_mb:.1f} MB)")
        print("-" * 80)
        
        # Test 1: Simple sequential loading
        print("1. Simple sequential loading:")
        start = time.time()
        simple_data = load_file_simple_optimized(test_file)
        simple_time = time.time() - start
        print(f"   Loaded {len(simple_data):,} items in {simple_time:.2f}s")
        
        # Test 2: UltraFastParallelLoader
        print("2. UltraFastParallelLoader:")
        start = time.time()
        dataset = UltraFastParallelLoader(
            parquet_files=[test_file],
            tokenizer=MockTokenizer(),
            max_length=25000,
            truncation='right',
            num_workers=4
        )
        parallel_time = time.time() - start
        print(f"   Loaded {len(dataset):,} items in {parallel_time:.2f}s")
        
        # Compare
        speedup = simple_time / parallel_time if parallel_time > 0 else 1.0
        if speedup > 1.0:
            print(f"   🚀 Parallel is {speedup:.2f}x FASTER!")
        else:
            print(f"   ⚠️  Parallel is {1/speedup:.2f}x slower (overhead)")
        
        # Verify correctness
        if len(simple_data) == len(dataset):
            print(f"   ✅ Size matches: {len(dataset):,} items")
        else:
            print(f"   ❌ Size mismatch: simple={len(simple_data)}, parallel={len(dataset)}")
        
        del dataset, simple_data  # Clean up memory

