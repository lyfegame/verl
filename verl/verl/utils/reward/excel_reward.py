import os
import json
from typing import Dict, Any, List, Optional, Tuple
import torch
import numpy as np


def excel_comparison_reward(
    data,
    tokenizer=None,
    reward_batch_size: int = 32,
    max_workers: int = 8,
    return_dict: bool = True,
    **kwargs
):
    """
    Custom reward function for Excel manipulation tasks.
    Compares the model's output Excel file with a reference file.
    
    Args:
        data: DataProto object containing prompts, responses, and metadata
        tokenizer: The tokenizer (optional)
        reward_batch_size: Batch size for processing
        max_workers: Maximum number of parallel workers
        return_dict: Whether to return results as a dictionary
        **kwargs: Additional keyword arguments
        
    Returns:
        Dictionary with 'reward_tensor' and 'reward_extra_info' if return_dict=True,
        otherwise just the reward tensor
    """
    
    # Import here to avoid circular imports
    try:
        from evaluation.ui.excel_sheet_comparison_ui import compare_excel_files
    except ImportError:
        print("Warning: Could not import compare_excel_files. Using dummy comparison.")
        def compare_excel_files(file1, file2, **params):
            # Dummy implementation for testing
            return file1 == file2, {} if file1 == file2 else {"error": "Files don't match"}
    
    # Get batch size
    batch_size = len(data)
    rewards = torch.zeros(batch_size, dtype=torch.float32)
    extra_info = {}
    
    # Get metadata from non_tensor_batch
    excel_metadata = data.non_tensor_batch.get('excel_metadata', None)
    
    if excel_metadata is None:
        print("Warning: No excel_metadata found in non_tensor_batch. Using default rewards.")
        if return_dict:
            return {
                "reward_tensor": rewards,
                "reward_extra_info": {"error": "No excel_metadata found"}
            }
        else:
            return rewards
    
    # Process each item in the batch
    for idx in range(batch_size):
        try:
            # Get metadata for this item
            metadata = excel_metadata[idx] if isinstance(excel_metadata[idx], dict) else excel_metadata[idx].item()
            
            # Extract file paths and verification parameters
            final_file = metadata.get('final_file', '')
            reference_file = metadata.get('reference_file', '')
            verifiers_dict = metadata.get('verifiers_dict', {
                'check_content': True,
                'check_conditional_formatting': True,
                'check_tables': True,
                'check_frozen_panes': True
            })
            task_id = metadata.get('task_id', f'task_{idx}')
            
            # NOTE: In a real implementation, you would generate the final_file
            # based on the model's response. For now, we assume it's pre-generated.
            # You might need to:
            # 1. Decode the response from data.batch["responses"][idx]
            # 2. Parse the response to extract Excel manipulation commands
            # 3. Apply those commands to the initial_file
            # 4. Save the result as final_file
            
            # Check if files exist
            if not os.path.exists(final_file):
                print(f"Warning: Final file not found for {task_id}: {final_file}")
                rewards[idx] = 0.0
                extra_info[task_id] = {"error": "Final file not found"}
                continue
                
            if not os.path.exists(reference_file):
                print(f"Warning: Reference file not found for {task_id}: {reference_file}")
                rewards[idx] = 0.0
                extra_info[task_id] = {"error": "Reference file not found"}
                continue
            
            # Compare the files
            print(f"Comparing files for {task_id}")
            print(f"  Final: {final_file}")
            print(f"  Reference: {reference_file}")
            print(f"  Params: {verifiers_dict}")
            
            is_identical, differences = compare_excel_files(
                final_file,
                reference_file,
                **verifiers_dict,
                file_name_1_local="final_file",
                file_name_2_local="reference_file"
            )
            
            # Assign reward based on comparison result
            if differences == {}:
                # Perfect match
                rewards[idx] = 1.0
                extra_info[task_id] = {"status": "perfect_match"}
            else:
                # Partial credit based on type of differences
                reward = calculate_partial_reward(differences, verifiers_dict)
                rewards[idx] = reward
                extra_info[task_id] = {
                    "status": "partial_match",
                    "differences": differences,
                    "partial_reward": reward
                }
                
            print(f"  Reward for {task_id}: {rewards[idx]}")
            
        except Exception as e:
            print(f"Error processing task {idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            rewards[idx] = 0.0
            extra_info[f"task_{idx}"] = {"error": str(e)}
    
    # Ensure rewards tensor has correct shape (batch_size, 1) for PPO
    rewards = rewards.unsqueeze(-1)
    
    if return_dict:
        return {
            "reward_tensor": rewards,
            "reward_extra_info": extra_info
        }
    else:
        return rewards


def calculate_partial_reward(differences: Dict[str, Any], verifiers_dict: Dict[str, bool]) -> float:
    """
    Calculate partial reward based on the types of differences found.
    
    Args:
        differences: Dictionary of differences between files
        verifiers_dict: Dictionary of what to check
        
    Returns:
        Partial reward score between 0 and 1
    """
    if not differences:
        return 1.0
        
    # Count number of checks enabled
    enabled_checks = sum(1 for check, enabled in verifiers_dict.items() if enabled)
    if enabled_checks == 0:
        return 0.0
    
    # Count number of passing checks
    passing_checks = enabled_checks
    
    # Check each type of difference
    if verifiers_dict.get('check_content', True) and 'content' in differences:
        passing_checks -= 1
        
    if verifiers_dict.get('check_conditional_formatting', True) and 'openpyxl' in differences:
        openpyxl_diff = differences.get('openpyxl', {})
        for sheet_diff in openpyxl_diff.values():
            if 'conditional_formatting' in sheet_diff:
                passing_checks -= 0.25  # Smaller penalty for formatting
                break
                
    if verifiers_dict.get('check_tables', True) and 'openpyxl' in differences:
        openpyxl_diff = differences.get('openpyxl', {})
        for sheet_diff in openpyxl_diff.values():
            if 'tables' in sheet_diff:
                passing_checks -= 0.5
                break
                
    if verifiers_dict.get('check_frozen_panes', True) and 'openpyxl' in differences:
        openpyxl_diff = differences.get('openpyxl', {})
        for sheet_diff in openpyxl_diff.values():
            if 'frozen_panes' in sheet_diff:
                passing_checks -= 0.25  # Smaller penalty
                break
    
    # Return normalized score
    return max(0.0, passing_checks / enabled_checks)


# For testing the reward function
if __name__ == "__main__":
    # Create dummy data for testing
    from verl import DataProto
    import numpy as np
    
    # Create test data
    test_metadata = np.array([{
        'task_id': 'test_task',
        'final_file': '/path/to/final.xlsx',
        'reference_file': '/path/to/reference.xlsx',
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
    
    result = excel_comparison_reward(data)
    print(f"Test result: {result}") 