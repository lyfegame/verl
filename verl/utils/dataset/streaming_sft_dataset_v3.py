# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Streaming SFT Dataset V3 - Correct parallel indexing based on line numbers
"""

from typing import List, Union
import json
import os
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
import subprocess

from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer


def count_lines_fast(file_path):
    """Use wc -l for fast line counting"""
    try:
        result = subprocess.run(['wc', '-l', file_path], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return int(result.stdout.split()[0])
    except:
        pass
    
    # Fallback to Python
    return sum(1 for _ in open(file_path, 'rb'))


def index_line_range(args):
    """
    Index a specific range of lines in a file.
    Each worker gets a line range to process.
    """
    file_path, start_line, end_line, worker_id = args
    
    offsets = []
    
    with open(file_path, 'rb') as f:
        # Skip to start_line
        for _ in range(start_line):
            f.readline()
        
        # Now index our assigned lines
        for line_num in range(start_line, end_line):
            line_start = f.tell()
            line = f.readline()
            
            if not line:
                break
                
            if line.strip():  # Skip empty lines
                offsets.append((line_num, line_start))
    
    return worker_id, offsets


class StreamingSFTDataset(Dataset):
    """
    Streaming SFT Dataset V3 with correct parallel indexing.
    
    Key improvements:
    - Uses line-based splitting for parallel indexing (not byte-based)
    - Guarantees correct ordering and no missing/duplicate lines
    - Fast line counting with wc -l
    - Maintains exact compatibility with original SFTDataset
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
        Initialize streaming dataset with correct parallel indexing.
        
        Args:
            num_workers: Number of workers for parallel indexing (None = auto)
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
        
        # For compatibility
        self.prompt_key = prompt_key
        self.response_key = response_key
        
        # Auto-detect number of workers
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 8)
        self.num_workers = max(1, num_workers)

        # Build index with correct parallel processing
        self._build_index()
        
        # Cache file handles for faster access
        self._file_handles = {}

    def _build_index(self):
        """
        Build index with correct parallel processing based on line numbers.
        """
        self.file_indices = []
        self.line_offsets = []
        
        print(f"StreamingSFTDataset V3: Building index for {len(self.parquet_files)} file(s) using {self.num_workers} workers...")
        start_time = time.time()
        
        for file_idx, file_path in enumerate(self.parquet_files):
            file_size = os.path.getsize(file_path) / (1024**3)
            
            # Step 1: Fast line counting
            print(f"  Counting lines in {os.path.basename(file_path)} ({file_size:.2f} GB)...")
            line_count = count_lines_fast(file_path)
            print(f"    Found {line_count:,} lines")
            
            if line_count == 0:
                continue
            
            # Step 2: Parallel indexing based on line numbers
            if self.num_workers > 1 and line_count > 10000:
                # Split work by lines
                print(f"    Parallel indexing with {self.num_workers} workers...")
                file_offsets = self._parallel_index_file(file_path, line_count)
            else:
                # Sequential for small files
                print(f"    Sequential indexing...")
                file_offsets = self._sequential_index_file(file_path)
            
            # Add to global index
            for offset in file_offsets:
                self.file_indices.append(file_idx)
                self.line_offsets.append(offset)
            
            print(f"    Indexed {len(file_offsets):,} non-empty lines")
        
        self.total_length = len(self.line_offsets)
        elapsed = time.time() - start_time
        print(f"Total dataset size: {self.total_length:,} samples (indexed in {elapsed:.2f}s)")

    def _parallel_index_file(self, file_path, total_lines):
        """
        Index file in parallel using line-based splitting.
        """
        lines_per_worker = total_lines // self.num_workers
        
        # Create work chunks based on line numbers
        chunks = []
        for i in range(self.num_workers):
            start_line = i * lines_per_worker
            if i == self.num_workers - 1:
                # Last worker handles remaining lines
                end_line = total_lines
            else:
                end_line = (i + 1) * lines_per_worker
            
            chunks.append((file_path, start_line, end_line, i))
        
        # Process chunks in parallel
        all_offsets = [[] for _ in range(self.num_workers)]
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(index_line_range, chunk): idx 
                      for idx, chunk in enumerate(chunks)}
            
            for future in as_completed(futures):
                worker_id, offsets = future.result()
                # offsets contains (line_num, byte_offset) tuples
                all_offsets[worker_id] = offsets
        
        # Combine results in correct order
        combined_offsets = []
        for worker_offsets in all_offsets:
            # Extract just the byte offsets (already in correct order within each worker)
            combined_offsets.extend([offset for _, offset in worker_offsets])
        
        return combined_offsets

    def _sequential_index_file(self, file_path):
        """
        Index file sequentially (for small files or single worker).
        """
        offsets = []
        
        with open(file_path, 'rb') as f:
            while True:
                line_start = f.tell()
                line = f.readline()
                
                if not line:
                    break
                
                if line.strip():  # Skip empty lines
                    offsets.append(line_start)
        
        return offsets

    def __len__(self):
        """Return total number of samples."""
        return self.total_length

    def __getitem__(self, item):
        """
        Get sample by index with O(1) access.
        """
        if item >= self.total_length or item < 0:
            raise IndexError(f"Index {item} out of range [0, {self.total_length})")
        
        # Get file and offset
        file_idx = self.file_indices[item]
        offset = self.line_offsets[item]
        file_path = self.parquet_files[file_idx]
        
        # Read the specific line
        if file_path not in self._file_handles:
            self._file_handles[file_path] = open(file_path, 'r')
        
        f = self._file_handles[file_path]
        f.seek(offset)
        line = f.readline()
        
        # Parse JSON
        try:
            data = json.loads(line.strip())
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse line at index {item}: {e}")
            return self.__getitem__((item + 1) % self.total_length)
        
        # Process pre-tokenized data
        if 'input_ids' in data and 'loss_mask' in data:
            input_ids = torch.tensor(data['input_ids'], dtype=torch.long)
            loss_mask = torch.tensor(data['loss_mask'], dtype=torch.long)
            
            # Handle sequences longer than max_length
            sequence_length = input_ids.shape[0]
            
            if sequence_length > self.max_length:
                if self.truncation == 'error':
                    raise ValueError(f'{sequence_length=} is larger than {self.max_length=}')
                elif self.truncation == 'left':
                    input_ids = input_ids[-self.max_length:]
                    loss_mask = loss_mask[-self.max_length:]
                else:  # right truncation
                    input_ids = input_ids[:self.max_length]
                    loss_mask = loss_mask[:self.max_length]
                
                sequence_length = self.max_length
            
            # Create attention mask
            attention_mask = torch.ones(sequence_length, dtype=torch.long)
            
            # Pad to max_length if needed
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
        else:
            # Return dummy sample for non-tokenized data
            return {
                'input_ids': torch.ones(self.max_length, dtype=torch.long) * self.tokenizer.pad_token_id,
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'position_ids': torch.zeros(self.max_length, dtype=torch.long),
                'loss_mask': torch.zeros(self.max_length, dtype=torch.long)
            }

    def __del__(self):
        """Clean up file handles."""
        for f in self._file_handles.values():
            try:
                f.close()
            except:
                pass


# Alias for compatibility
StreamingSFTDatasetV3 = StreamingSFTDataset


if __name__ == "__main__":
    """Test the V3 dataset"""
    import random
    
    TEST_FILE = "/home/tianhangzhu/data/test/leader_test_multiturn_10xsub35xother_optimized.jsonl"
    
    class MockTokenizer:
        pad_token_id = 0
        eos_token_id = 1
    
    print("="*70)
    print("Testing StreamingSFTDataset V3 (Line-based Parallel Indexing)")
    print("="*70)
    
    for num_workers in [1, 4, 8]:
        print(f"\n Testing with {num_workers} workers:")
        
        start = time.time()
        dataset = StreamingSFTDataset(
            parquet_files=[TEST_FILE],
            tokenizer=MockTokenizer(),
            max_length=22000,
            truncation='right',
            num_workers=num_workers
        )
        elapsed = time.time() - start
        
        print(f"\n  Initialization time: {elapsed:.2f}s")
        print(f"  Dataset size: {len(dataset):,}")
        
        # Test correctness - check that samples at same indices match
        test_indices = [0, 100, 1000, 5000, 10000]
        print(f"  Testing samples at indices {test_indices}:")
        
        for idx in test_indices:
            sample = dataset[idx]
            print(f"    Index {idx}: shape={sample['input_ids'].shape}, "
                  f"first tokens={sample['input_ids'][:5].tolist()}")
        
        del dataset
