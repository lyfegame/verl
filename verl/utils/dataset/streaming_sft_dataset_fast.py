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
Fast Streaming SFT Dataset - Minimal indexing for faster startup
Only counts lines, doesn't build full offset index
"""

from typing import List, Union
import json
import os
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer


class StreamingSFTDatasetFast(Dataset):
    """
    Fast streaming dataset that only counts lines instead of building full index.
    Trades random access performance for much faster initialization.
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer,
                 prompt_key='prompt',
                 prompt_dict_keys=None,
                 response_key='response',
                 response_dict_keys=None,
                 max_length=1024,
                 truncation='error'):
        """
        Initialize streaming dataset with minimal indexing.
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

        # Just count lines - MUCH faster than full indexing
        self._count_lines()
        
        # Cache for frequently accessed items
        self._cache = {}
        self._cache_size = 100  # Cache last 100 accessed items

    def _count_lines(self):
        """
        Count lines in each file using the fastest method available.
        """
        self.file_lengths = []
        self.cumulative_lengths = [0]
        
        print(f"StreamingSFTDatasetFast: Quick counting lines in {len(self.parquet_files)} file(s)...")
        
        for file_idx, file_path in enumerate(self.parquet_files):
            file_size = os.path.getsize(file_path) / (1024**3)
            
            # Try to use wc -l command if available (much faster)
            try:
                import subprocess
                result = subprocess.run(['wc', '-l', file_path], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    line_count = int(result.stdout.split()[0])
                    print(f"  {os.path.basename(file_path)}: {line_count:,} lines, {file_size:.2f} GB (fast count)")
                else:
                    raise Exception("wc failed")
            except:
                # Fallback to Python counting (slower but portable)
                print(f"  Counting: {file_path} (Python fallback)...")
                line_count = sum(1 for _ in open(file_path, 'rb'))
                print(f"    {line_count:,} lines, {file_size:.2f} GB")
            
            self.file_lengths.append(line_count)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + line_count)
        
        self.total_length = self.cumulative_lengths[-1]
        print(f"Total dataset size: {self.total_length:,} samples")

    def __len__(self):
        """Return total number of samples."""
        return self.total_length

    def _find_file_and_line(self, idx):
        """Find which file and line number for given index."""
        # Binary search for efficiency
        left, right = 0, len(self.cumulative_lengths) - 1
        while left < right:
            mid = (left + right) // 2
            if self.cumulative_lengths[mid] <= idx < self.cumulative_lengths[mid + 1]:
                return mid, idx - self.cumulative_lengths[mid]
            elif idx < self.cumulative_lengths[mid]:
                right = mid
            else:
                left = mid + 1
        return left, idx - self.cumulative_lengths[left]

    def __getitem__(self, item):
        """
        Get a sample by index. Uses caching for frequently accessed items.
        For uncached items, reads sequentially from the start of the file chunk.
        """
        if item >= self.total_length or item < 0:
            raise IndexError(f"Index {item} out of range [0, {self.total_length})")
        
        # Check cache first
        if item in self._cache:
            return self._cache[item]
        
        # Find which file and line
        file_idx, line_idx = self._find_file_and_line(item)
        file_path = self.parquet_files[file_idx]
        
        # Read the specific line (need to read from start of file)
        # This is slower than offset-based seeking but avoids the indexing time
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i == line_idx:
                    line = line.strip()
                    if not line:
                        return self.__getitem__((item + 1) % self.total_length)
                    
                    try:
                        data = json.loads(line)
                        result = self._process_sample(data)
                        
                        # Add to cache
                        if len(self._cache) >= self._cache_size:
                            # Remove oldest item (simple FIFO)
                            self._cache.pop(next(iter(self._cache)))
                        self._cache[item] = result
                        
                        return result
                    except json.JSONDecodeError:
                        return self.__getitem__((item + 1) % self.total_length)
        
        # Should not reach here
        raise RuntimeError(f"Could not find line {line_idx} in file {file_idx}")

    def _process_sample(self, data):
        """Process a single sample from JSON data."""
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


class StreamingSFTDatasetNoIndex(Dataset):
    """
    Ultra-fast version that doesn't count lines at all.
    Just returns a fixed large number for length.
    Best for when you don't need exact dataset size.
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
                 estimated_samples_per_gb=7000):  # Rough estimate
        """
        Initialize with no indexing at all - instant startup.
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
        
        # Estimate size based on file size (no reading at all)
        total_size_gb = sum(os.path.getsize(f) / (1024**3) for f in parquet_files)
        self.estimated_length = int(total_size_gb * estimated_samples_per_gb)
        
        print(f"StreamingSFTDatasetNoIndex: {len(parquet_files)} files, ~{total_size_gb:.2f} GB")
        print(f"Estimated samples: ~{self.estimated_length:,} (no indexing performed)")
        
        # Keep file handles open for faster access
        self._file_handles = {}
        self._current_positions = {}

    def __len__(self):
        """Return estimated length."""
        return self.estimated_length

    def __getitem__(self, item):
        """
        Simple modulo-based access - doesn't guarantee unique samples
        but is extremely fast.
        """
        # Simple file selection based on modulo
        file_idx = item % len(self.parquet_files)
        file_path = self.parquet_files[file_idx]
        
        # Open file if not already open
        if file_path not in self._file_handles:
            self._file_handles[file_path] = open(file_path, 'r')
            self._current_positions[file_path] = 0
        
        f = self._file_handles[file_path]
        
        # Try to read next line from current position
        line = f.readline()
        if not line:
            # Reached end, restart from beginning
            f.seek(0)
            line = f.readline()
        
        if line:
            try:
                data = json.loads(line.strip())
                return self._process_sample(data)
            except:
                # On error, just return next item
                return self.__getitem__((item + 1) % self.estimated_length)
        
        # Shouldn't reach here
        return self._dummy_sample()

    def _process_sample(self, data):
        """Same as in Fast version"""
        # [Same implementation as StreamingSFTDatasetFast._process_sample]
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
            return self._dummy_sample()

    def _dummy_sample(self):
        """Return a dummy sample."""
        return {
            'input_ids': torch.ones(self.max_length, dtype=torch.long) * self.tokenizer.pad_token_id,
            'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
            'position_ids': torch.zeros(self.max_length, dtype=torch.long),
            'loss_mask': torch.zeros(self.max_length, dtype=torch.long)
        }

    def __del__(self):
        """Close file handles."""
        for f in self._file_handles.values():
            try:
                f.close()
            except:
                pass
