import json
import os
from typing import Dict, List, Any, Optional
import pandas as pd
import openpyxl
from openpyxl.utils import get_column_letter
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import io
import base64
import torch


class ExcelTaskDataset(Dataset):
    """Dataset for Excel manipulation tasks that loads initial Excel files and formats them for model input."""
    
    def __init__(self, data_files: List[str], tokenizer, processor, config):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.tasks = []
        
        # Ensure data_files is a list
        if isinstance(data_files, str):
            data_files = [data_files]
        
        # Load all tasks from JSONL files
        for file_path in data_files:
            with open(file_path, 'r') as f:
                for line in f:
                    task = json.loads(line.strip())
                    self.tasks.append(task)
        
        print(f"Loaded {len(self.tasks)} tasks from {len(data_files)} files")
        
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        task = self.tasks[idx]
        
        # Extract Excel content from initial file
        excel_content = self._extract_excel_content(task['initial_file'])
        
        # Create the prompt
        system_prompt = """You are an AI assistant that helps users manipulate Excel spreadsheets. You will be given the current state of a spreadsheet and a task to perform. Your goal is to provide clear instructions or formulas to accomplish the task."""
        
        user_prompt = f"""{excel_content}

Task: {task['user_query']}"""
        
        # Format messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        tokenized = self.tokenizer(
            prompt,
            padding=False,
            truncation=True,
            max_length=self.config.max_prompt_length,
            return_tensors="pt"
        )
        
        # Flatten tensors to remove extra dimension
        for key in tokenized.keys():
            if isinstance(tokenized[key], torch.Tensor):
                tokenized[key] = tokenized[key].squeeze(0)
        
        # Create non-tensor data with metadata for reward computation
        non_tensor_data = {
            'excel_metadata': np.array([{
                'task_id': task['task_id'],
                'final_file': task.get('final_file', ''),
                'reference_file': task.get('reference_file', ''),
                'verifiers_dict': task.get('verifiers_dict', {}),
                'user_query': task['user_query'],
                'initial_file': task['initial_file']
            }], dtype=object)
        }
        
        # Add required fields for VERL
        tokenized['raw_prompt'] = np.array([prompt], dtype=object)
        tokenized['raw_prompt_ids'] = np.array([tokenized['input_ids'].tolist()], dtype=object) 
        
        return {
            **tokenized,
            **non_tensor_data
        }
    
    def _extract_excel_content(self, excel_path: str, max_rows: int = 20, max_cols: int = 20, max_cells: int = 400) -> str:
        """Extract content from Excel file similar to the JavaScript implementation."""
        try:
            # Load workbook
            wb = openpyxl.load_workbook(excel_path, data_only=False)
            content_parts = ["The following is some of the data in the spreadsheet. Use this data to help you complete the task."]
            
            for sheet_idx, sheet_name in enumerate(wb.sheetnames):
                sheet = wb[sheet_name]
                
                # Get used range
                max_row = sheet.max_row
                max_col = sheet.max_column
                
                # Check if we should include full data
                total_cells = max_row * max_col
                if total_cells > max_cells:
                    data_rows = min(max_row, max_rows)
                    data_cols = min(max_col, max_cols)
                else:
                    data_rows = max_row
                    data_cols = max_col
                
                # Extract table information
                table_info = []
                for table in sheet.tables.values():
                    table_info.append(f"{table.displayName}: {table.ref}")
                tables_str = ", ".join(table_info) if table_info else "None"
                
                # Extract chart information  
                charts_str = f"{len(sheet._charts)} charts" if hasattr(sheet, '_charts') else "0 charts"
                
                # Build content for this sheet
                content_parts.append(f"""
-----------------
Sheet Index: {sheet_idx}
Sheet Name: "{sheet_name}"
Total Rows: {max_row}
Total Columns: {max_col}
Existing Tables: {tables_str}
Existing Charts: {charts_str}
- Formulas are shown in parentheses if they exist
- Not all rows and columns may be displayed below

Data (in CSV format):""")
                
                # Extract data with formulas
                csv_data = self._format_sheet_as_csv(sheet, data_rows, data_cols)
                content_parts.append(csv_data)
                
            return "\n".join(content_parts)
            
        except Exception as e:
            print(f"Error loading Excel file {excel_path}: {e}")
            return f"Error loading Excel file: {str(e)}"
    
    def _format_sheet_as_csv(self, sheet, max_rows: int, max_cols: int) -> str:
        """Format sheet data as CSV, including formulas in parentheses."""
        rows = []
        
        # Add column headers (A, B, C, etc.)
        headers = [""] + [get_column_letter(col) for col in range(1, max_cols + 1)]
        rows.append(",".join(headers))
        
        # Add data rows
        for row_idx in range(1, max_rows + 1):
            row_data = [str(row_idx)]  # Row number
            
            for col_idx in range(1, max_cols + 1):
                cell = sheet.cell(row=row_idx, column=col_idx)
                
                # Get cell value
                value = cell.value if cell.value is not None else ""
                
                # Check if cell has formula
                if hasattr(cell, '_value') and isinstance(cell._value, str) and cell._value.startswith('='):
                    # Show both formula and computed value
                    cell_str = f"{value} ({cell._value})"
                else:
                    cell_str = str(value)
                
                # Handle special characters in CSV
                if ',' in cell_str or '\n' in cell_str or '"' in cell_str:
                    cell_str = '"' + cell_str.replace('"', '""') + '"'
                    
                row_data.append(cell_str)
                
            rows.append(",".join(row_data))
            
        return "\n".join(rows) 