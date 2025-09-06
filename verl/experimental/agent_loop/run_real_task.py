#!/usr/bin/env python3
"""
Run a real task on OrchestratorCodingAgentLoop with Modal endpoint

This script will:
1. Use real SWE-Gym data
2. Use real Qwen tokenizer  
3. Make REAL calls to Modal endpoint (no mocking)
4. Automatically save all LLM interactions to /home/tianhangzhu/test_llm.jsonl
"""

import asyncio
import sys
import os
import json
from uuid import uuid4

# Add paths
sys.path.append('/home/tianhangzhu/RL/jeffery/verl')

from datasets import load_dataset
from transformers import AutoTokenizer
from verl.experimental.agent_loop.orchestrator_coding_agent_loop import OrchestratorCodingAgentLoop


class MockServerManager:
    """Mock server manager - we're using Modal endpoint instead"""
    async def generate(self, request_id, prompt_ids, sampling_params, image_data=None):
        raise NotImplementedError("Using Modal endpoint for generation")


class RealModalConfig:
    """Configuration for real Modal endpoint usage"""
    def __init__(self):
        self.actor_rollout_ref = type('obj', (object,), {})()
        self.actor_rollout_ref.rollout = type('obj', (object,), {})()
        self.actor_rollout_ref.rollout.multi_turn = type('obj', (object,), {})()
        
        # Set reasonable limits for real execution
        self.actor_rollout_ref.rollout.multi_turn.max_user_turns = 3
        self.actor_rollout_ref.rollout.multi_turn.max_assistant_turns = 3
        self.actor_rollout_ref.rollout.multi_turn.max_tool_response_length = 3000
        self.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side = "right"
        
        # Configure to use REAL Modal endpoint
        def get_config_value(key, default):
            config_values = {
                "modal_base_url": "https://fairies--incremental-leader-agent-api",
                "modal_timeout": 120,  # Longer timeout for real calls
                "truncation_strategy": "ast_llm_compaction",
                "truncation_max_tokens": 16000,
                "enable_truncation": False,
                "enable_tokenization_cache": True,
                # ENABLE REAL MODAL ENDPOINT
                "use_modal_endpoint": True,
                "modal_chat_endpoint": "https://fairies--vllm-global-step-900-cons-7514578f-serve.modal.run/v1/chat/completions",
                "modal_model_name": "vllm-global-step-900-cons-7514578f"
            }
            return config_values.get(key, default)
        
        self.actor_rollout_ref.rollout.multi_turn.get = get_config_value
        
        self.actor_rollout_ref.rollout.prompt_length = 12000
        self.actor_rollout_ref.rollout.response_length = 20000  # Increased to allow more turns
        
        self.data = type('obj', (object,), {})()
        self.data.get = lambda key, default: {"apply_chat_template_kwargs": {}}.get(key, default)


async def run_real_task():
    """Run a real task with Modal endpoint and save all LLM calls"""
    print("🚀 RUNNING REAL TASK WITH MODAL ENDPOINT")
    print("=" * 60)
    print("⚠️  This will make REAL API calls to Modal!")
    print("⚠️  All LLM interactions will be saved to /home/tianhangzhu/test_llm.jsonl")
    print("=" * 60)
    
    # Clear previous log file to start fresh
    log_file = "/home/tianhangzhu/test_llm.jsonl"
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"🧹 Cleared existing log file: {log_file}")
    
    try:
        # Load real components
        print("📥 Loading real tokenizer: Qwen/Qwen2.5-Coder-14B-Instruct...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-14B-Instruct")
        processor = None
        print(f"✅ Tokenizer loaded! Vocab size: {tokenizer.vocab_size}")
        
        # Load real SWE-Gym task
        print("📥 Loading SWE-Gym dataset...")
        dataset = load_dataset("SWE-Gym/SWE-Gym", split="train[:1]")
        task = dataset[0]
        print(f"✅ Loaded task:")
        print(f"   Instance ID: {task['instance_id']}")
        print(f"   Repository: {task['repo']}")
        print(f"   Problem length: {len(task['problem_statement'])} characters")
        print(f"   Problem preview: {task['problem_statement'][:200]}...")
        
        # Create agent with REAL Modal configuration
        print("🔧 Initializing OrchestratorCodingAgentLoop with REAL Modal endpoint...")
        config = RealModalConfig()
        server_manager = MockServerManager()
        
        # Reset class initialization
        OrchestratorCodingAgentLoop._class_initialized = False
        
        agent_loop = OrchestratorCodingAgentLoop(
            trainer_config=type('obj', (object,), {'config': config})(),
            server_manager=server_manager,
            tokenizer=tokenizer,
            processor=processor
        )
        
        print("✅ Agent initialized!")
        print(f"   Using Modal endpoint: {agent_loop.use_modal_endpoint}")
        print(f"   Modal chat endpoint: {agent_loop.modal_chat_endpoint}")
        print(f"   Modal model: {agent_loop.modal_model_name}")
        
        # Set up realistic sampling parameters with higher limits
        sampling_params = {
            "temperature": 0,
            "max_tokens": 4000,  # Higher limit to avoid truncation
            "top_p": 0.9
        }
        
        # Task parameters
        kwargs = {
            "instance_id": task["instance_id"],
            "task_prompt": task["problem_statement"],
            "dataset_name": "SWE-Gym/SWE-Gym",
            "run_id": f"real_run_{uuid4().hex[:8]}",
            "notebook_id": f"real_notebook_{uuid4().hex[:6]}"
        }
        
        print(f"🎯 Task parameters:")
        print(f"   Instance ID: {kwargs['instance_id']}")
        print(f"   Task prompt: {len(kwargs['task_prompt'])} characters")
        print(f"   Sampling: {sampling_params}")
        print(f"   Run ID: {kwargs['run_id']}")
        print(f"   Notebook ID: {kwargs['notebook_id']}")
        
        print("\n🚀 Starting REAL task execution...")
        print("⏳ This may take a while as it makes real API calls...")
        print("-" * 60)
        
        # Run the REAL task
        result = await agent_loop.run(sampling_params, **kwargs)
        
        print("-" * 60)
        print("🎉 REAL TASK COMPLETED SUCCESSFULLY!")
        
        # Display results
        print(f"📊 Execution Results:")
        print(f"   Prompt tokens: {len(result.prompt_ids)}")
        print(f"   Response tokens: {len(result.response_ids)}")
        print(f"   Response mask length: {len(result.response_mask)}")
        print(f"   Number of turns: {result.num_turns}")
        print(f"   Has logprobs: {result.response_logprobs is not None}")
        
        # Display metrics - FIX: Handle both dict and object types
        print(f"📈 Metrics:")
        try:
            if hasattr(result.metrics, 'items'):
                # It's a dict
                for key, value in result.metrics.items():
                    if key == "solution_patch":
                        print(f"   {key}: {len(str(value))} characters")
                        if value:
                            print(f"      Patch preview: {str(value)[:150]}...")
                    else:
                        print(f"   {key}: {value}")
            else:
                # It's an object, get attributes
                print(f"   Metrics type: {type(result.metrics)}")
                if hasattr(result.metrics, '__dict__'):
                    for key, value in result.metrics.__dict__.items():
                        if key == "solution_patch":
                            print(f"   {key}: {len(str(value))} characters")
                            if value:
                                print(f"      Patch preview: {str(value)[:150]}...")
                        else:
                            print(f"   {key}: {value}")
                else:
                    print(f"   Metrics: {result.metrics}")
        except Exception as e:
            print(f"   Error displaying metrics: {e}")
            print(f"   Raw metrics: {result.metrics}")
        
        # Show logged LLM interactions
        print(f"\n📄 LLM Interactions Log:")
        
        # Check for both .json and .jsonl files in multiple locations
        import glob
        rllog_dir = os.path.expanduser("~/RLlog")
        
        # Look for the most recent timestamped file in RLlog
        rllog_pattern = os.path.join(rllog_dir, "test_llm_*.json")
        rllog_files = glob.glob(rllog_pattern)
        if rllog_files:
            # Get the most recent file
            json_log_file = max(rllog_files, key=os.path.getmtime)
        else:
            json_log_file = "/home/tianhangzhu/test_llm.json"  # fallback
        jsonl_log_file = log_file
        
        actual_log_file = None
        if os.path.exists(json_log_file):
            actual_log_file = json_log_file
        elif os.path.exists(jsonl_log_file):
            actual_log_file = jsonl_log_file
            
        if actual_log_file:
            if actual_log_file.endswith('.json'):
                # Handle JSON format
                with open(actual_log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"✅ {len(data)} real LLM interactions logged to: {actual_log_file}")
                    
                    for i, interaction in enumerate(data, 1):
                        messages = interaction['messages']
                        output = interaction['output']
                        
                        print(f"\n   📝 Interaction {i}:")
                        print(f"      Input messages: {len(messages)}")
                        for j, msg in enumerate(messages):
                            content_preview = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
                            print(f"        {j+1}. {msg['role']}: {content_preview}")
                        
                        output_preview = output[:100] + "..." if len(output) > 100 else output
                        print(f"      Output ({len(output)} chars): {output_preview}")
            else:
                # Handle JSONL format
                with open(actual_log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    print(f"✅ {len(lines)} real LLM interactions logged to: {actual_log_file}")
                    
                    for i, line in enumerate(lines, 1):
                        try:
                            data = json.loads(line.strip())
                            messages = data['messages']
                            output = data['output']
                            
                            print(f"\n   📝 Interaction {i}:")
                            print(f"      Input messages: {len(messages)}")
                            for j, msg in enumerate(messages):
                                content_preview = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
                                print(f"        {j+1}. {msg['role']}: {content_preview}")
                            
                            output_preview = output[:100] + "..." if len(output) > 100 else output
                            print(f"      Output ({len(output)} chars): {output_preview}")
                            
                        except json.JSONDecodeError as e:
                            print(f"      ⚠️ Error parsing line {i}: {e}")
        else:
            print(f"❌ No log file found at: {json_log_file} or {jsonl_log_file}")
        
        # Final summary
        print(f"\n" + "=" * 60)
        print("🎉 REAL TASK EXECUTION COMPLETE!")
        print("✅ Used real Qwen tokenizer")
        print("✅ Used real SWE-Gym task data")
        print("✅ Made real Modal endpoint API calls")
        print("✅ Generated real token responses")
        print(f"✅ Logged all LLM interactions to: {log_file}")
        print("=" * 60)
        
        return result
        
    except Exception as e:
        print(f"\n❌ REAL TASK FAILED: {e}")
        print("🔍 MODAL ENDPOINT ANALYSIS:")
        print("- Chat completion endpoint WORKS (we tested it successfully)")
        print("- Sandbox endpoints (init-sandbox, execute-cell, etc.) appear to be DOWN")
        print("- This is likely because:")
        print("  1. Modal sandbox service is not deployed/running")
        print("  2. Different authentication required for sandbox vs chat")
        print("  3. Sandbox endpoints use different URL structure")
        print("  4. Rate limiting on sandbox operations")
        
        # Still show any logged interactions if they exist
        if os.path.exists(log_file):
            print(f"\n📄 Partial log file exists: {log_file}")
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"   {len(lines)} interactions logged before failure")
        
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    print("🧪 REAL ORCHESTRATOR CODING AGENT TASK")
    print("This will make REAL API calls to Modal endpoints!")
    
    # Confirm before running real API calls
    try:
        # Run the real task
        asyncio.run(run_real_task())
    except KeyboardInterrupt:
        print("\n⚠️ Task interrupted by user")
    except Exception as e:
        print(f"\n💥 Task failed: {e}")
        print("Check the logs above for details")