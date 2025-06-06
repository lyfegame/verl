#!/usr/bin/env python3
"""Test script to validate Excel dataset and reward function integration with VERL."""

import torch
import numpy as np
from verl import DataProto
from verl.utils import hf_tokenizer

def test_excel_dataset():
    """Test the Excel dataset functionality."""
    print("Testing Excel Dataset...")
    
    # Create dummy config
    class Config:
        max_prompt_len = 1024
    
    # Load tokenizer (you can change to your model)
    tokenizer = hf_tokenizer("gpt2")
    
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
    import json
    test_file = "/tmp/test_excel_data.jsonl"
    with open(test_file, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    # Test dataset loading
    try:
        from verl.utils.dataset.excel_dataset import ExcelTaskDataset
        dataset = ExcelTaskDataset([test_file], tokenizer, None, Config())
        print(f"✓ Dataset created successfully with {len(dataset)} items")
        
        # Test getting an item
        item = dataset[0]
        print(f"✓ Dataset item keys: {list(item.keys())}")
        
        # Check if excel_metadata is present
        if 'excel_metadata' in item:
            print(f"✓ Excel metadata found: {item['excel_metadata'][0].keys()}")
        else:
            print("✗ Excel metadata not found!")
            
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()


def test_excel_reward():
    """Test the Excel reward function."""
    print("\nTesting Excel Reward Function...")
    
    try:
        from verl.utils.reward.excel_reward import excel_comparison_reward
        
        # Create test DataProto
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
        
        data = DataProto.from_dict(
            tensors={},
            non_tensors={'excel_metadata': test_metadata},
            meta_info={}
        )
        
        # Test reward computation
        result = excel_comparison_reward(data)
        print(f"✓ Reward function executed")
        print(f"  - Reward tensor shape: {result['reward_tensor'].shape}")
        print(f"  - Extra info keys: {list(result['reward_extra_info'].keys())}")
        
    except Exception as e:
        print(f"✗ Reward function test failed: {e}")
        import traceback
        traceback.print_exc()


def test_dataproto_integration():
    """Test DataProto integration with collate_fn."""
    print("\nTesting DataProto Integration...")
    
    try:
        from verl.utils.dataset.rl_dataset import collate_fn
        
        # Create test batch
        batch = [
            {
                'input_ids': torch.tensor([1, 2, 3, 4]),
                'attention_mask': torch.tensor([1, 1, 1, 0]),
                'raw_prompt': np.array(['Test prompt 1'], dtype=object),
                'raw_prompt_ids': np.array([[1, 2, 3, 4]], dtype=object),
                'excel_metadata': np.array([{'task_id': 'task_1'}], dtype=object)
            },
            {
                'input_ids': torch.tensor([5, 6, 7, 8]),
                'attention_mask': torch.tensor([1, 1, 1, 1]),
                'raw_prompt': np.array(['Test prompt 2'], dtype=object),
                'raw_prompt_ids': np.array([[5, 6, 7, 8]], dtype=object),
                'excel_metadata': np.array([{'task_id': 'task_2'}], dtype=object)
            }
        ]
        
        # Test collation
        collated = collate_fn(batch)
        print(f"✓ Collation successful")
        print(f"  - Tensor keys: {[k for k in collated.keys() if isinstance(collated[k], torch.Tensor)]}")
        print(f"  - Non-tensor keys: {[k for k in collated.keys() if not isinstance(collated[k], torch.Tensor)]}")
        
        # Convert to DataProto
        data_proto = DataProto.from_single_dict(collated)
        print(f"✓ DataProto created")
        print(f"  - Batch size: {len(data_proto)}")
        print(f"  - Non-tensor batch keys: {list(data_proto.non_tensor_batch.keys())}")
        
    except Exception as e:
        print(f"✗ DataProto integration test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 60)
    print("Excel Integration Tests for VERL")
    print("=" * 60)
    
    test_excel_dataset()
    test_excel_reward()
    test_dataproto_integration()
    
    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60) 