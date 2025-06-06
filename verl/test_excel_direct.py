#!/usr/bin/env python3
"""Direct test of Excel modules without importing verl package."""

import sys
import os
import json
import torch
import numpy as np

# Add paths directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'verl', 'utils', 'dataset'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'verl', 'utils', 'reward'))

def test_excel_dataset():
    """Test the Excel dataset functionality."""
    print("Testing Excel Dataset...")
    
    # Import directly
    import excel_dataset
    
    # Create dummy config
    class Config:
        max_prompt_len = 1024
    
    # Create a simple tokenizer mock
    class MockTokenizer:
        def __init__(self):
            self.eos_token_id = 50256
            self.pad_token_id = 50256
            
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "System: " + messages[0]["content"] + "\nUser: " + messages[1]["content"] + "\nAssistant:"
            
        def __call__(self, text, padding=False, truncation=True, max_length=1024, return_tensors="pt"):
            # Simple mock tokenization
            tokens = list(range(min(len(text), max_length)))
            result = {
                "input_ids": torch.tensor([tokens]),
                "attention_mask": torch.ones(len(tokens), dtype=torch.long).unsqueeze(0)
            }
            return result
            
        def decode(self, ids, skip_special_tokens=True):
            return "decoded text"
    
    tokenizer = MockTokenizer()
    
    # Create test data
    test_data = [{
        "task_id": "test_task_1",
        "user_query": "Add a header to column F",
        "initial_file": "/tmp/test_initial.xlsx",
        "final_file": "/tmp/test_final.xlsx", 
        "reference_file": "/tmp/test_reference.xlsx",
        "verifiers_dict": {
            "check_content": True,
            "check_conditional_formatting": True,
            "check_tables": True,
            "check_frozen_panes": True
        }
    }]
    
    # Save test data to JSONL
    test_file = "/tmp/test_excel_data.jsonl"
    with open(test_file, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    # Test dataset loading
    try:
        dataset = excel_dataset.ExcelTaskDataset([test_file], tokenizer, None, Config())
        print(f"✓ Dataset created successfully with {len(dataset)} items")
        
        # Test getting an item
        item = dataset[0]
        print(f"✓ Dataset item keys: {list(item.keys())}")
        
        # Check if excel_metadata is present
        if 'excel_metadata' in item:
            metadata = item['excel_metadata'][0]
            print(f"✓ Excel metadata found with keys: {list(metadata.keys())}")
            print(f"  - task_id: {metadata['task_id']}")
        else:
            print("✗ Excel metadata not found!")
            
        # Check tensor shapes
        print(f"✓ input_ids shape: {item['input_ids'].shape}")
        print(f"✓ attention_mask shape: {item['attention_mask'].shape}")
        
        # Check non-tensor data
        print(f"✓ raw_prompt type: {type(item['raw_prompt'])}")
        print(f"✓ raw_prompt_ids type: {type(item['raw_prompt_ids'])}")
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()


def test_excel_reward():
    """Test the Excel reward function."""
    print("\nTesting Excel Reward Function...")
    
    # Import directly
    import excel_reward
    
    try:
        # Create a simple DataProto mock
        class MockDataProto:
            def __init__(self, metadata):
                self.non_tensor_batch = {'excel_metadata': metadata}
                
            def __len__(self):
                return len(self.non_tensor_batch['excel_metadata'])
        
        # Create test metadata
        test_metadata = np.array([{
            'task_id': 'test_task',
            'final_file': '/tmp/test_final.xlsx',
            'reference_file': '/tmp/test_reference.xlsx',
            'verifiers_dict': {
                'check_content': True,
                'check_conditional_formatting': True,
                'check_tables': True,
                'check_frozen_panes': True
            }
        }], dtype=object)
        
        data = MockDataProto(test_metadata)
        
        # Test reward computation
        result = excel_reward.excel_comparison_reward(data)
        print(f"✓ Reward function executed")
        print(f"  - Reward tensor shape: {result['reward_tensor'].shape}")
        print(f"  - Reward tensor dtype: {result['reward_tensor'].dtype}")
        print(f"  - Extra info keys: {list(result['reward_extra_info'].keys())}")
        
        # Check the reward values
        print(f"  - Reward values: {result['reward_tensor'].squeeze().tolist()}")
        
    except Exception as e:
        print(f"✗ Reward function test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 60)
    print("Direct Excel Module Tests")
    print("=" * 60)
    
    test_excel_dataset()
    test_excel_reward()
    
    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60) 