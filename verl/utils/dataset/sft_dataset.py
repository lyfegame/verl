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
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

from typing import List, Union
import json
import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer


class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset
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
        assert truncation in ['error', 'left', 'right']
        self.truncation = truncation

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key
        self.response_key = response_key
        self.max_length = max_length

        self._read_files_and_tokenize()
    def _read_files_and_tokenize(self):
        # Load JSONL data
        data = []
        for file_path in self.parquet_files:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        self.data = data
        self.input_ids = []
        self.loss_masks = []
        
        for item in data:
            self.input_ids.append(item['input_ids'])
            self.loss_masks.append(item['loss_mask'])
        
        return

    def __len__(self):
        if hasattr(self, 'data'):
            return len(self.data)
        return len(self.prompts)

    def __getitem__(self, item):
        # If we're using the JSON data with pre-tokenized inputs
        input_ids = torch.tensor(self.input_ids[item], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        loss_mask = torch.tensor(self.loss_masks[item], dtype=torch.long)
        
        # Ensure the sequence is the right length

        sequence_length = input_ids.shape[0]
        assert sequence_length < self.max_length, f'{sequence_length=} is larger than {self.max_length=}'
        
        padded_input_ids = torch.ones(size=(self.max_length - sequence_length,),
                                    dtype=input_ids.dtype) * self.tokenizer.pad_token_id
        padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)
        padded_loss_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

        input_ids = torch.cat((input_ids, padded_input_ids))
        attention_mask = torch.cat((attention_mask, padded_attention_mask))
        loss_mask = torch.cat((loss_mask, padded_loss_mask))

        position_ids = compute_position_id_with_mask(attention_mask)


        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask
        }
        

if __name__ == '__main__':
    from transformers import AutoTokenizer
    local_model_path = "FundamentalResearchLabs/xlam-hf60ktools-sft"

    tokenizer = AutoTokenizer.from_pretrained(local_model_path, device_map="cuda", local_files_only=True)
       # Set pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load data from the distilled dataset
    dataset = SFTDataset(parquet_files=["/home/ubuntu/sharedusmidwest1/tianhangzhu/data/preposttrain/distill_new_automator/processed_distilled_train_only_data_all_ntasks674_n4_max_length_16384_size_3211_train.jsonl"], 
                         tokenizer=tokenizer,
                         max_length=3000,
                         truncation='right')


    print("\nExample 0:")
    sample = dataset[0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: shape={v.shape}, dtype={v.dtype}")

    print(tokenizer.decode(sample['input_ids']))