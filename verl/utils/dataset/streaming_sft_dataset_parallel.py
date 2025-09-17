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
Parallel Streaming SFT Dataset - Uses multiple CPUs for faster indexing
"""

from typing import List, Union, Dict, Tuple
import json
import os
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import mmap
import time

from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer


def count_lines_chunk(args):
    """Count lines in a chunk of a file (for parallel processing)"""
    file_path, start_byte, end_byte, chunk_id = args
    
    line_count = 0
    with open(file_path, 'rb') as f:
        f.seek(start_byte)
        
        # If not starting at beginning, skip to next newline
        if start_byte > 0:
            f.readline()
        
        while f.tell() < end_byte:
            line = f.readline()
            if not line:
                break
            line_count += 1
    
    return chunk_id, line_count


def index_file_chunk(args):
    """Build line offset index for a chunk of a file"""
    file_path, start_byte, end_byte, chunk_id = args
    
    offsets = []
    with open(file_path, 'rb') as f:
        f.seek(start_byte)
        
        # If not starting at beginning, skip to next newline
        if start_byte > 0:
            f.readline()
        
        while f.tell() < end_byte:
            line_start = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():  # Only index non-empty lines
                offsets.append(line_start)
    
    return chunk_id, offsets


def index_single_file(file_path, num_workers=4):
    """Index a single file using multiple workers"""
    file_size = os.path.getsize(file_path)
    
    if file_size < 10 * 1024 * 1024:  # Less than 10MB, don't parallelize
        offsets = []
        with open(file_path, 'rb') as f:
            while True:
                line_start = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(line_start)
        return offsets
    
    # Split file into chunks for parallel processing
    chunk_size = file_size // num_workers
    chunks = []
    for i in range(num_workers):
        start = i * chunk_size
        end = file_size if i == num_workers - 1 else (i + 1) * chunk_size
        chunks.append((file_path, start, end, i))
    
    # Process chunks in parallel
    all_offsets = [[] for _ in range(num_workers)]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(index_file_chunk, chunk): idx 
                  for idx, chunk in enumerate(chunks)}
        
        for future in as_completed(futures):
            chunk_id, offsets = future.result()
            all_offsets[chunk_id] = offsets
    
    # Combine results in order
    combined_offsets = []
    for offsets in all_offsets:
        combined_offsets.extend(offsets)
    
    return combined_offsets


def count_lines_fast(file_path):
    """Ultra-fast line counting using mmap and parallel processing"""
    try:
        # Try wc -l first (fastest)
        import subprocess
        result = subprocess.run(['wc', '-l', file_path], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return int(result.stdout.split()[0])
    except:
        pass
    
    # Fallback to mmap-based counting
    try:
        with open(file_path, 'r+b') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped:
                lines = 0
                while mmapped.readline():
                    lines += 1
                return lines
    except:
        # Final fallback
        return sum(1 for _ in open(file_path, 'rb'))


class StreamingSFTDatasetParallel(Dataset):
    """
    Parallel indexing version of StreamingSFTDataset.
    Uses multiple CPUs to build index faster for large files.
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
        Initialize with parallel indexing.
        
        Args:
            num_workers: Number of CPU workers for indexing (None = auto)
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
            num_workers = min(mp.cpu_count(), 8, len(parquet_files) * 2)
        self.num_workers = max(1, num_workers)
        
        # For compatibility
        self.prompt_key = prompt_key
        self.response_key = response_key

        # Build index in parallel
        self._build_index_parallel()
        
        # Cache file handles for faster access
        self._file_handles = {}

    def _build_index_parallel(self):
        """Build index using parallel processing"""
        print(f"StreamingSFTDatasetParallel: Indexing {len(self.parquet_files)} file(s) using {self.num_workers} workers...")
        start_time = time.time()
        
        if len(self.parquet_files) == 1:
            # Single file - parallelize within file
            self._build_single_file_parallel()
        else:
            # Multiple files - process files in parallel
            self._build_multi_file_parallel()
        
        elapsed = time.time() - start_time
        print(f"Indexing completed in {elapsed:.2f}s ({self.total_length:,} samples)")

    def _build_single_file_parallel(self):
        """Build index for a single large file using parallel chunks"""
        file_path = self.parquet_files[0]
        file_size = os.path.getsize(file_path) / (1024**3)
        print(f"  Parallel indexing single file: {file_path} ({file_size:.2f} GB)")
        
        # For single file, store offsets directly
        self.use_simple_index = True
        self.single_file = file_path
        
        # Count lines first (fast)
        line_count = count_lines_fast(file_path)
        print(f"    Found {line_count:,} lines")
        
        # Build offset index in parallel
        self.line_offsets = index_single_file(file_path, self.num_workers)
        self.total_length = len(self.line_offsets)
        
        if self.total_length != line_count:
            print(f"    Note: {line_count - self.total_length} empty lines skipped")

    def _build_multi_file_parallel(self):
        """Build index for multiple files in parallel"""
        self.use_simple_index = False
        self.file_indices = []
        self.line_offsets = []
        self.cumulative_lengths = [0]
        
        # Process all files in parallel
        file_results = {}
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}
            
            for idx, file_path in enumerate(self.parquet_files):
                # Submit both line counting and offset building
                futures[executor.submit(self._index_file, file_path)] = (idx, file_path)
            
            for future in as_completed(futures):
                idx, file_path = futures[future]
                offsets = future.result()
                file_results[idx] = (file_path, offsets)
                
                file_size = os.path.getsize(file_path) / (1024**3)
                print(f"    {os.path.basename(file_path)}: {len(offsets):,} lines, {file_size:.2f} GB")
        
        # Combine results in order
        for idx in sorted(file_results.keys()):
            file_path, offsets = file_results[idx]
            
            for offset in offsets:
                self.file_indices.append(idx)
                self.line_offsets.append(offset)
            
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(offsets))
        
        self.total_length = len(self.line_offsets)

    def _index_file(self, file_path):
        """Index a single file (for parallel execution)"""
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
        """Get sample by index with O(1) access time"""
        if item >= self.total_length or item < 0:
            raise IndexError(f"Index {item} out of range [0, {self.total_length})")
        
        # Get file and offset
        if self.use_simple_index:
            file_path = self.single_file
            offset = self.line_offsets[item]
        else:
            file_idx = self.file_indices[item]
            file_path = self.parquet_files[file_idx]
            offset = self.line_offsets[item]
        
        # Read the specific line
        if file_path not in self._file_handles:
            self._file_handles[file_path] = open(file_path, 'r')
        
        f = self._file_handles[file_path]
        f.seek(offset)
        line = f.readline()
        
        # Parse and process
        try:
            data = json.loads(line.strip())
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse line at index {item}: {e}")
            return self.__getitem__((item + 1) % self.total_length)
        
        return self._process_sample(data)

    def _process_sample(self, data):
        """Process a single sample"""
        if 'input_ids' in data and 'loss_mask' in data:
            input_ids = torch.tensor(data['input_ids'], dtype=torch.long)
            loss_mask = torch.tensor(data['loss_mask'], dtype=torch.long)
            
            sequence_length = input_ids.shape[0]
            
            if sequence_length > self.max_length:
                if self.truncation == 'error':
                    raise ValueError(f'{sequence_length=} is larger than {self.max_length=}')
                elif self.truncation == 'left':
                    input_ids = input_ids[-self.max_length:]
                    loss_mask = loss_mask[-self.max_length:]
                else:
                    input_ids = input_ids[:self.max_length]
                    loss_mask = loss_mask[:self.max_length]
                
                sequence_length = self.max_length
            
            attention_mask = torch.ones(sequence_length, dtype=torch.long)
            
            if sequence_length < self.max_length:
                pad_length = self.max_length - sequence_length
                
                padded_input_ids = torch.ones(pad_length, dtype=input_ids.dtype) * self.tokenizer.pad_token_id
                input_ids = torch.cat((input_ids, padded_input_ids))
                
                padded_attention_mask = torch.zeros(pad_length, dtype=attention_mask.dtype)
                attention_mask = torch.cat((attention_mask, padded_attention_mask))
                
                padded_loss_mask = torch.zeros(pad_length, dtype=loss_mask.dtype)
                loss_mask = torch.cat((loss_mask, padded_loss_mask))
            
            position_ids = compute_position_id_with_mask(attention_mask)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'loss_mask': loss_mask
            }
        else:
            # Return dummy sample
            return {
                'input_ids': torch.ones(self.max_length, dtype=torch.long) * self.tokenizer.pad_token_id,
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'position_ids': torch.zeros(self.max_length, dtype=torch.long),
                'loss_mask': torch.zeros(self.max_length, dtype=torch.long)
            }

    def __del__(self):
        """Clean up file handles"""
        for f in self._file_handles.values():
            try:
                f.close()
            except:
                pass


if __name__ == "__main__":
    """Test the parallel indexing dataset"""
    import time
    
    TEST_FILE = "/home/tianhangzhu/data/test/leader_test_multiturn_10xsub35xother_optimized.jsonl"
    
    class MockTokenizer:
        pad_token_id = 0
        eos_token_id = 1
    
    print("Testing StreamingSFTDatasetParallel")
    print("="*60)
    
    # Test with different worker counts
    for num_workers in [1, 4, 8, None]:
        print(f"\nTesting with {num_workers if num_workers else 'auto'} workers:")
        
        start = time.time()
        dataset = StreamingSFTDatasetParallel(
            parquet_files=[TEST_FILE],
            tokenizer=MockTokenizer(),
            max_length=22000,
            truncation='right',
            num_workers=num_workers
        )
        elapsed = time.time() - start
        
        print(f"  Dataset size: {len(dataset):,}")
        print(f"  Total time: {elapsed:.2f}s")
        
        # Test access
        sample = dataset[0]
        print(f"  Sample 0 shape: {sample['input_ids'].shape}")
        
        del dataset  # Clean up



