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
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import re
from contextlib import nullcontext

import hydra
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch import nn, optim
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import Dataset, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

import verl.utils.hdfs_io as hdfs_io
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, get_checkpoint_tracker_filename
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.dataset.streaming_sft_dataset_final import StreamingSFTDataset as StreamingSFTDatasetV2
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.logger import log_with_rank
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from verl.utils.tracking import Tracking
from verl.utils.ulysses import (
    gather_outputs_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))

import subprocess
import sys
from pathlib import Path


def consolidate_fsdp_checkpoint(checkpoint_path: str) -> str:
    """Consolidate FSDP checkpoint to HuggingFace format."""
    checkpoint_path = Path(checkpoint_path).absolute()
    
    # Check if already consolidated
    hf_dir = checkpoint_path / "huggingface"
    if hf_dir.exists():
        # Check if it has actual model files
        model_files = list(hf_dir.glob("*.safetensors")) + list(hf_dir.glob("*.bin"))
        if model_files:
            print("✓ Checkpoint already consolidated (found model files in huggingface/)")
            return str(hf_dir)
    
    print("Detected FSDP checkpoint format. Starting consolidation...")
    
    # Create a temporary directory for consolidated model
    consolidated_dir = checkpoint_path.parent / f"{checkpoint_path.name}_consolidated"
    consolidated_dir.mkdir(exist_ok=True)
    
    # Run the FSDP model merger
    cmd = [
        sys.executable, "-m", "verl.model_merger", "merge",
        "--backend", "fsdp",
        "--local_dir", str(checkpoint_path),
        "--target_dir", str(consolidated_dir)
    ]
    
    print("Consolidating FSDP checkpoint to HuggingFace format...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"✗ FSDP consolidation failed: {result.stderr}")
        raise Exception(f"FSDP consolidation failed: {result.stderr}")
    
    print(f"✓ FSDP checkpoint consolidated to: {consolidated_dir}")
    return str(consolidated_dir)

def extract_step(path):
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


class FSDPSFTTrainer:
    def __init__(
        self,
        config,
        device_mesh: DeviceMesh,
        ulysses_device_mesh: DeviceMesh,
        tokenizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        if device_mesh.get_rank() == 0:
            print(f"[DEBUG] FSDPSFTTrainer.__init__ started")
            print(f"[DEBUG] Train dataset size: {len(train_dataset)}")
            print(f"[DEBUG] Val dataset size: {len(val_dataset)}")
            
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.tokenizer = tokenizer
        if self.config.data.chat_template is not None:
            raise ValueError("Apply Chat template from config is not supported yet.")

        # normalize dp size
        if self.device_mesh.get_rank() == 0:
            print(f"[DEBUG] Normalizing batch sizes...")
        self._normalize_config_bsz()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(self.config, "ulysses_sequence_parallel_size", 1)
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)
        if self.device_mesh.get_rank() == 0:
            print(f"[DEBUG] Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}")
            print(f"[DEBUG] Using remove padding: {self.use_remove_padding}")

        if self.device_mesh.get_rank() == 0:
            print(f"[DEBUG] Building dataloaders...")
        self._build_dataloader(train_dataset, val_dataset)

        # Initialize resume-related variables
        self.resume_global_step = 0

        # build model
        if self.device_mesh.get_rank() == 0:
            print(f"[DEBUG] Building model and optimizer...")
        self._build_model_optimizer()

        # Initialize checkpoint manager
        if self.device_mesh.get_rank() == 0:
            print(f"[DEBUG] Initializing checkpoint manager...")
        self._init_checkpoint_manager()

        if self.device_mesh.get_rank() == 0:
            print(f"[DEBUG] Loading checkpoint (if exists)...")
        self.load_checkpoint()

        if self.device_mesh.get_rank() == 0:
            print(f"[DEBUG] FSDPSFTTrainer initialization complete")
            print(self.config)
        self.device_name = self.config.trainer.device

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        assert self.config.data.train_batch_size % dp_size == 0, (
            f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"
        )

        self.config.data.train_batch_size //= dp_size

        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset, val_dataset):
        # build dataset
        config = self.config
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # If doing SP, we need to use the local rank and size
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank("dp")
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f"Using SP rank {rank} and size {world_size} for data distribution")
                print("Each SP rank gets different data, but the same data WITHIN the same rank")
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        # Set pin_memory_device when pin_memory is enabled.
        device_name = get_device_name()

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True, seed=config.trainer.seed
        )
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            pin_memory_device=device_name,
        )

        self.val_sampler = DistributedSampler(
            self.val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            pin_memory_device=device_name,
        )

    def _build_model_optimizer(self):
        if self.device_mesh.get_rank() == 0:
            print(f"[DEBUG] _build_model_optimizer started")
            print(f"[DEBUG] Model path: {self.config.model.partial_pretrain}")
            
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)
        
        if self.device_mesh.get_rank() == 0:
            print(f"[DEBUG] Model copied to local: {local_model_path}")

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("[DEBUG] Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)
        
        if self.device_mesh.get_rank() == 0:
            print(f"[DEBUG] Loading model config with dtype: {torch_dtype}")
            
        # load config first
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        self.model_config = config
        if hasattr(self.model_config, "max_position_embeddings"):
            self.model_config.max_position_embeddings = max(
                self.model_config.max_position_embeddings, self.config.data.max_length
            )
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        # This may be very large
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context():
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(model=self.model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)

            # Apply Liger kernel if use_liger is enabled
            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=self.model)

            if self.config.model.get("lora_rank", 0) > 0:
                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))
                self.model = self.model.to(torch_dtype)

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            self.fsdp_model = FSDP(
                self.model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
            )

            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": True,
            }
            full_state = self.model.state_dict()
            apply_fsdp2(self.model, fsdp_kwargs, self.config.model.fsdp_config)
            fsdp2_load_full_state_dict(self.model, full_state, self.device_mesh, cpu_offload)
            self.fsdp_model = self.model
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        self.optimizer = optim.AdamW(
            self.fsdp_model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )

        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            print(
                f"Number of steps/epoch {self.steps_per_epoch}, number of epochs "
                f"{self.config.trainer.total_epochs}, total number of steps {self.total_steps}"
            )

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")

    def _compute_loss_and_backward(self, batch, do_backward=True):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1

        # Move inputs to GPU and prepare loss mask
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        loss_mask = batch.pop("loss_mask")[:, :-1].reshape(-1).to(self.device_name)
        loss_fct = nn.CrossEntropyLoss(reduction="none")

        # Context manager for sequence parallel if needed
        context = self.sharding_manager if use_sp else nullcontext()
        with context, torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            if not use_sp:
                # Standard forward pass without sequence parallel
                labels = input_ids[:, 1:].contiguous()
                output = self.fsdp_model(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                )
                logits = output.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels.contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                loss = loss * loss_mask.to(loss.device)
            else:
                # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
                # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
                # 1. All SP ranks will receive the *SAME* batch
                # 2. Different SP groups will receive *DIFFERENT* batches
                # This is implemented by the DistributedSampler

                batch_size, seqlen = input_ids.shape
                # Remove padding
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # Unpad position_ids to align rotary
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

                # Pad and slice inputs for sequence parallelism
                input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size()
                )
                # For computing loss
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size()
                )
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # Forward pass
                output = self.fsdp_model(
                    input_ids=input_ids_rmpad_sliced,
                    attention_mask=None,  # Not needed with flash attention varlen
                    position_ids=position_ids_rmpad_padded,
                    use_cache=False,
                )

                # Compute loss locally then aggregate
                logits_rmpad = output.logits.squeeze(0)
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                # Gather and unpad for sequence parallelism
                loss = gather_outputs_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # This is the loss collected from all ulysses ranks
                full_loss = pad_input(
                    hidden_states=loss.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )
                full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss
                full_loss = full_loss.reshape(-1)
                loss_mask = loss_mask.to(full_loss.device)
                loss = full_loss * loss_mask

            valid_token_this_rank = torch.sum(loss_mask)

            if self.config.data.balance_dp_token:
                torch.distributed.all_reduce(valid_token_this_rank)
                dp_size = self.ulysses_device_mesh.size("dp") if use_sp else torch.distributed.get_world_size()
            else:
                dp_size = 1

            loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size

            if do_backward:
                loss.backward()
            return loss

    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()

        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)

        self.optimizer.zero_grad()

        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        for micro_batch in micro_batches:
            loss = self._compute_loss_and_backward(batch=micro_batch) / n_micro_batches
            step_loss += loss.item()

        if self.config.model.strategy == "fsdp":
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == "fsdp2":
            grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
        else:
            raise NotImplementedError(f"not implement {self.config.model.strategy}")

        log_gpu_memory_usage("Before optimizer step", logger=logger)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        log_gpu_memory_usage("After optimizer step", logger=logger)

        self.lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage("After offload weights", logger=logger)

        step_loss = torch.tensor(step_loss).to(self.device_name)
        if is_cuda_available:
            torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        elif is_npu_available:
            torch.distributed.all_reduce(step_loss)
            step_loss /= self.device_mesh.size(0)
        return {"train/loss": step_loss.detach().item(), "train/lr(1e-3)": lr * 1e3}

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss = self._compute_loss_and_backward(batch, do_backward=False)
            if is_cuda_available:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
            elif is_npu_available:
                torch.distributed.all_reduce(loss)
                loss /= self.device_mesh.size(0)
        return loss

    def save_checkpoint(self, step):
        """Save checkpoint using FSDPCheckpointManager with improved tracking and permission handling"""
        from verl.utils.fs import local_mkdir_safe
        import subprocess
        import stat
        import time
        import signal

        # Retry logic for handling I/O errors
        max_retries = 3
        retry_delay = 15  # seconds
        timeout_seconds = 30 * 60  # 30 minutes timeout per attempt
        
        for attempt in range(max_retries):
            try:
                # Define timeout handler
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Checkpoint save timed out after {timeout_seconds} seconds")
                
                # Set up timeout for this attempt
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)
                
                # Determine checkpoint path
                local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")

                if self.device_mesh.get_rank() == 0:
                    print(f"Saving checkpoint to: {local_global_step_folder} (attempt {attempt + 1}/{max_retries}, timeout: {timeout_seconds}s)")

                # Pre-create directories with proper permissions (before checkpoint manager)
                if self.device_mesh.get_rank() == 0:
                    try:
                        # First, ensure parent directory exists with proper permissions
                        parent_dir = self.config.trainer.default_local_dir
                        if not os.path.exists(parent_dir):
                            # Try to create with normal permissions first
                            try:
                                os.makedirs(parent_dir, mode=0o755, exist_ok=True)
                            except PermissionError:
                                # If permission denied, try with sudo (only if explicitly enabled)
                                if getattr(self.config.trainer, "use_sudo_for_checkpoint", False):
                                    print(f"Permission denied, attempting to create {parent_dir} with sudo...")
                                    subprocess.run(["sudo", "mkdir", "-p", parent_dir], check=True)
                                    subprocess.run(["sudo", "chmod", "755", parent_dir], check=True)
                                    # Change ownership to current user
                                    import pwd
                                    username = pwd.getpwuid(os.getuid()).pw_name
                                    subprocess.run(["sudo", "chown", f"{username}:{username}", parent_dir], check=True)
                                else:
                                    raise PermissionError(
                                        f"Cannot create {parent_dir}. Set trainer.use_sudo_for_checkpoint=true to use sudo"
                                    )

                        # Create checkpoint directory
                        if not os.path.exists(local_global_step_folder):
                            try:
                                os.makedirs(local_global_step_folder, mode=0o755, exist_ok=True)
                            except PermissionError:
                                if getattr(self.config.trainer, "use_sudo_for_checkpoint", False):
                                    print(f"Permission denied, attempting to create {local_global_step_folder} with sudo...")
                                    subprocess.run(["sudo", "mkdir", "-p", local_global_step_folder], check=True)
                                    subprocess.run(["sudo", "chmod", "755", local_global_step_folder], check=True)
                                    # Change ownership to current user
                                    import pwd
                                    username = pwd.getpwuid(os.getuid()).pw_name
                                    subprocess.run(["sudo", "chown", "-R", f"{username}:{username}", local_global_step_folder], check=True)
                                else:
                                    raise
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"Error creating checkpoint directory: {e}")
                            print(f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            print(f"Failed to create checkpoint directory after {attempt + 1} attempts: {e}")
                            raise

                # Synchronize before saving
                torch.distributed.barrier()

                # Get max checkpoints to keep
                max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)

                # Use checkpoint manager to save
                self.checkpoint_manager.save_checkpoint(
                    local_path=local_global_step_folder, global_step=step, max_ckpt_to_keep=max_ckpt_to_keep
                )

                # Save dataloader state with permission handling
                if self.device_mesh.get_rank() == 0:
                    dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")

                    try:
                        # Use StatefulDataLoader's built-in state dict functionality
                        dataloader_state_dict = self.train_dataloader.state_dict()
                        
                        # Save with explicit permission handling
                        temp_dataloader_path = dataloader_local_path + ".tmp"
                        torch.save(dataloader_state_dict, temp_dataloader_path)
                        
                        # Set proper permissions on the file
                        os.chmod(temp_dataloader_path, 0o644)
                        
                        # Atomic rename
                        os.rename(temp_dataloader_path, dataloader_local_path)
                        print(f"Saved dataloader state to: {dataloader_local_path}")
                    except PermissionError as e:
                        print(f"Permission error saving dataloader state: {e}")
                        if getattr(self.config.trainer, "use_sudo_for_checkpoint", False):
                            # Save to temp location and move with sudo
                            temp_path = f"/tmp/dataloader_state_{step}.pt"
                            torch.save(dataloader_state_dict, temp_path)
                            subprocess.run(["sudo", "mv", temp_path, dataloader_local_path], check=True)
                            subprocess.run(["sudo", "chmod", "644", dataloader_local_path], check=True)
                            print(f"Saved dataloader state with sudo to: {dataloader_local_path}")
                        else:
                            raise

                    # Update latest checkpoint tracker with atomic write and permission handling
                    try:
                        tracker_file = get_checkpoint_tracker_filename(self.config.trainer.default_local_dir)
                        temp_tracker_file = tracker_file + ".tmp"
                        
                        # Write to temp file
                        with open(temp_tracker_file, "w") as f:
                            f.write(str(step))
                        
                        # Set proper permissions
                        os.chmod(temp_tracker_file, 0o644)
                        
                        # Atomic rename
                        os.rename(temp_tracker_file, tracker_file)
                        print(f"Updated checkpoint tracker: {tracker_file}")
                    except PermissionError as e:
                        print(f"Permission error updating tracker: {e}")
                        if getattr(self.config.trainer, "use_sudo_for_checkpoint", False):
                            temp_path = f"/tmp/checkpoint_tracker_{step}.txt"
                            with open(temp_path, "w") as f:
                                f.write(str(step))
                            subprocess.run(["sudo", "mv", temp_path, tracker_file], check=True)
                            subprocess.run(["sudo", "chmod", "644", tracker_file], check=True)
                            print(f"Updated checkpoint tracker with sudo: {tracker_file}")
                        else:
                            raise

                # Copy to HDFS if configured
                if self.device_mesh.get_rank() == 0 and getattr(self.config.trainer, "default_hdfs_dir", None):
                    try:
                        hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
                        hdfs_io.copy(src=local_global_step_folder, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)
                    except Exception as e:
                        print(f"Warning: Failed to copy to HDFS: {e}")
                        # Continue even if HDFS copy fails
                if self.device_mesh.get_rank() == 0:
                    consolidate_fsdp_checkpoint(local_global_step_folder)
                    
                torch.distributed.barrier()
                
                # Successfully saved checkpoint, break out of retry loop
                if self.device_mesh.get_rank() == 0:
                    print(f"Successfully saved checkpoint at step {step}")
                break
                
            except TimeoutError as e:
                # Cancel the alarm
                signal.alarm(0)
                if attempt < max_retries - 1:
                    if self.device_mesh.get_rank() == 0:
                        print(f"Checkpoint save timed out: {e}")
                        print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    if self.device_mesh.get_rank() == 0:
                        print(f"Checkpoint save timed out after {max_retries} attempts: {e}")
                    raise
                
            except Exception as e:
                # Cancel the alarm
                signal.alarm(0)
                if attempt < max_retries - 1:
                    if self.device_mesh.get_rank() == 0:
                        print(f"Error during checkpoint save: {e}")
                        print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    if self.device_mesh.get_rank() == 0:
                        print(f"Failed to save checkpoint after {max_retries} attempts: {e}")
                    raise
            finally:
                # Always cancel the alarm to prevent it from firing later
                signal.alarm(0)

    def _init_checkpoint_manager(self):
        """Initialize checkpoint manager with proper configuration"""
        # Get checkpoint configuration from config, with defaults
        checkpoint_config = getattr(self.config.trainer, "checkpoint", {})

        # Set default values if not specified
        save_contents = checkpoint_config.get("save_contents", ["model", "optimizer", "extra"])
        load_contents = checkpoint_config.get("load_contents", save_contents)

        # Create checkpoint config dict
        checkpoint_config_dict = {
            "load_contents": load_contents,
            "save_contents": save_contents,
        }

        # Convert to DictConfig for compatibility
        checkpoint_config_dict = DictConfig(checkpoint_config_dict)

        # Initialize checkpoint manager
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.fsdp_model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.tokenizer,
            checkpoint_config=checkpoint_config_dict,
        )

    def load_checkpoint(self):
        # Determine resume path based on configuration
        checkpoint_path = self._determine_resume_path()

        if checkpoint_path is None:
            return 0

        # extract resume step from checkpoint path
        resume_step = extract_step(checkpoint_path)
        if resume_step is None:
            log_with_rank(
                f"Warning: Could not extract step number from {checkpoint_path}, starting from step 0",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                level=logging.WARNING,
                log_only_rank_0=True,
            )
            return 0
        self.resume_global_step = resume_step

        # Use checkpoint manager to load model state
        self.checkpoint_manager.load_checkpoint(checkpoint_path)
        log_with_rank(
            f"Successfully loaded model checkpoint from {checkpoint_path} (step {resume_step})",
            logger=logger,
            rank=self.device_mesh.get_rank(),
            log_only_rank_0=True,
        )

        # Always load dataloader state for StatefulDataLoader
        self._load_dataloader_state(checkpoint_path)

        return resume_step

    def _load_dataloader_state(self, checkpoint_path: str):
        """Load dataloader state from checkpoint"""
        dataloader_path = os.path.join(checkpoint_path, "data.pt")

        if os.path.exists(dataloader_path):
            # Use StatefulDataLoader's built-in state dict functionality
            dataloader_state_dict = torch.load(dataloader_path, map_location="cpu", weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)

            log_with_rank(
                f"Successfully loaded dataloader state from {dataloader_path}",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                log_only_rank_0=True,
            )

        else:
            log_with_rank(
                f"Warning: No dataloader state found at {dataloader_path}, will start from scratch",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                level=logging.WARNING,
                log_only_rank_0=True,
            )

    def _determine_resume_path(self):
        """Determine the path to resume from based on resume_mode configuration"""
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)

        if resume_mode == "disable":
            return None
        elif resume_mode == "auto":
            if resume_from_path is not None:
                assert os.path.exists(resume_from_path), (
                    "resume_from_path must be null or an existing path when resume_mode is 'auto'"
                )
                assert "global_step_" in resume_from_path, "resume_from_path must specify the global_steps"
                return resume_from_path
            # Try to find the latest checkpoint in the default directory
            return self._find_latest_checkpoint()
        elif resume_mode == "resume_path":
            assert os.path.exists(resume_from_path), (
                "resume_from_path must be an existing path when resume_mode is 'resume_path'"
            )
            assert "global_step_" in resume_from_path, "resume_from_path must specify the global_steps"
            return resume_from_path
        else:
            raise ValueError(f"Invalid resume_mode: {resume_mode}. Must be 'auto', 'disable', or 'resume_path'")

    def _find_latest_checkpoint(self):
        """Find the latest checkpoint in the default local directory"""
        checkpoint_dir = self.config.trainer.default_local_dir

        if not os.path.exists(checkpoint_dir):
            return None

        latest_checkpoint = find_latest_ckpt_path(checkpoint_dir)

        if latest_checkpoint and self.device_mesh.get_rank() == 0:
            step_num = extract_step(latest_checkpoint)
            print(f"Found latest checkpoint: {latest_checkpoint} (step {step_num})")

        return latest_checkpoint

    def fit(self):
        rank = self.device_mesh.get_rank()
        
        if rank == 0:
            print(f"[DEBUG] fit() started - initializing training")
            print(f"[DEBUG] Train dataloader length: {len(self.train_dataloader)}")
            print(f"[DEBUG] Total epochs: {self.config.trainer.total_epochs}")

        # TODO: add a unified tracking
        if rank == 0:
            print(f"[DEBUG] Initializing tracking...")
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )
            print(f"[DEBUG] Tracking initialized")

        global_step = self.resume_global_step  # Start from resumed step
        if rank == 0:
            print(f"[DEBUG] Starting from global step: {global_step}")
            
        last_valid_metric = None
        # compute the total training steps.
        # the total training steps in SFT is mainly for early exit
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        log_with_rank(
            f"[DEBUG] Total training steps: {self.total_training_steps}",
            logger=logger,
            rank=self.device_mesh.get_rank(),
            log_only_rank_0=True,
        )

        # With StatefulDataLoader, we don't need to manually calculate epochs and steps
        # The dataloader will automatically resume from where it left off
        if global_step > 0:
            log_with_rank(
                f"StatefulDataLoader will automatically resume from global step: {global_step}",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                log_only_rank_0=True,
            )

        # Calculate which epoch we're starting from for sampler.set_epoch()
        start_epoch = global_step // self.steps_per_epoch

        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            if rank == 0:
                print(f"[DEBUG] Starting epoch {epoch + 1}/{self.config.trainer.total_epochs}")
                
            self.train_sampler.set_epoch(epoch=epoch)

            for step_in_epoch, data in enumerate(
                tqdm(
                    self.train_dataloader,
                    initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                    disable=rank != 0,
                )
            ):
                global_step += 1
                
                # Add detailed debug logging every N steps
                if rank == 0 and (global_step == 1 or global_step % 10 == 0):
                    print(f"[DEBUG] Processing step {global_step}/{self.total_training_steps}")
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / (1024**3)
                        reserved = torch.cuda.memory_reserved() / (1024**3)
                        print(f"[DEBUG] GPU memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
                
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).to(self.device_name)
                metric = self.training_step(data)
                if rank == 0:
                    tracking.log(data=metric, step=global_step)
                    
                    # Log metrics every N steps
                    if global_step % 10 == 0:
                        print(f"[DEBUG] Step {global_step} metrics: {metric}")

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.config.trainer.test_freq == 0
                is_save_step = global_step % self.config.trainer.save_freq == 0

                # early exit or validation step
                if is_last_step or (self.config.trainer.test_freq > 0 and is_valid_step):
                    # Perform validation
                    val_losses = []
                    for val_data in self.val_dataloader:
                        val_data = TensorDict(val_data, batch_size=self.config.data.micro_batch_size_per_gpu).to(
                            self.device_name
                        )
                        val_loss = self.validation_step(val_data)
                        val_losses.append(val_loss)
                    if rank == 0:
                        val_loss = torch.mean(torch.stack(val_losses))
                        metric = {"val/loss": val_loss.detach().item()}
                        tracking.log(data=metric, step=global_step)
                        last_valid_metric = metric
                    torch.distributed.barrier()

                if is_last_step or (self.config.trainer.save_freq > 0 and is_save_step):
                    self.save_checkpoint(step=global_step)

                if is_last_step:
                    if rank == 0:
                        print(f"Final validation metrics: {last_valid_metric}")
                    return


def run_sft(config):
    print(f"[DEBUG] Starting run_sft with config")
    print(f"[DEBUG] Model: {config.model.partial_pretrain}")
    print(f"[DEBUG] Max length: {config.data.get('max_length', 'not set')}")
    print(f"[DEBUG] Train batch size: {config.data.get('train_batch_size', 'not set')}")
    print(f"[DEBUG] Micro batch size: {config.data.get('micro_batch_size_per_gpu', 'not set')}")
    
    device_name = get_device_name()
    print(f"[DEBUG] Device: {device_name}")
    
    local_rank, rank, world_size = initialize_global_process_group()
    print(f"[DEBUG] Process initialized - Rank: {rank}, World size: {world_size}, Local rank: {local_rank}")

    print(f"[DEBUG] Initializing device mesh...")
    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )
    print(f"[DEBUG] Device mesh initialized - DP size: {dp_size}, Ulysses SP size: {config.ulysses_sequence_parallel_size}")
    
    # build tokenizer and datasets first
    from verl.utils import hf_tokenizer

    print(f"[DEBUG] Loading tokenizer from {config.model.partial_pretrain}...")
    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
    print(f"[DEBUG] Tokenizer loaded successfully")
    
    print(f"[DEBUG] Creating training dataset...")
    train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
    print(f"[DEBUG] Training dataset created - Size: {len(train_dataset)}")
    
    print(f"[DEBUG] Creating validation dataset...")
    val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)
    print(f"[DEBUG] Validation dataset created - Size: {len(val_dataset)}")

    print(f"[DEBUG] Initializing FSDPSFTTrainer...")
    trainer = FSDPSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    print(f"[DEBUG] FSDPSFTTrainer initialized successfully")

    print(f"[DEBUG] Starting training...")
    trainer.fit()
    print(f"[DEBUG] Training completed")

    print(f"[DEBUG] Destroying process group...")
    destroy_global_process_group()
    print(f"[DEBUG] Process group destroyed - run_sft completed")


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    run_sft(config)


def create_sft_dataset(data_paths, data_config, tokenizer):
    """Create a dataset."""
    import multiprocessing as mp
    
    print(f"[DEBUG] Creating SFT dataset from {len(data_paths)} files")
    
    # Auto-detect optimal number of workers for parallel indexing
    total_cpus = mp.cpu_count()
    
    # Check if user specified num_workers in config, otherwise auto-detect
    if data_config.get('num_workers', None) is not None:
        num_workers = data_config.get('num_workers')
        print(f"[DEBUG] Using user-specified {num_workers} workers for dataset indexing")
    else:
        # Auto-detect: use cpus-5 but cap at 32 for efficiency
        num_workers = min(64, max(1, total_cpus - 5))  # Cap at 32, leave 5 CPUs free
        print(f"[DEBUG] System has {total_cpus} CPUs, auto-using {num_workers} workers for dataset indexing")
    
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_type
        print(f"[DEBUG] Loading custom dataset class from {data_config.custom_cls.path}")
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    # Then check if multi-turn dataset should be used
    elif data_config.get("multiturn", {}).get("enable", False):
        print(f"[DEBUG] Using MultiTurnSFTDataset")
        dataset_cls = MultiTurnSFTDataset
    # Always use StreamingSFTDatasetV2 for regular datasets (memory efficient)
    else:
        dataset_cls = StreamingSFTDatasetV2
        print(f"[DEBUG] Using StreamingSFTDatasetV2 for memory-efficient loading")
        print(f"[DEBUG] Max length: {data_config.get('max_length', 1024)}")
        print(f"[DEBUG] Truncation: {data_config.get('truncation', 'error')}")

    # Create datasets based on the selected class
    print(f"[DEBUG] Creating dataset with class: {dataset_cls.__name__}")
    
    if dataset_cls == StreamingSFTDatasetV2:
        # StreamingSFTDatasetV2 with parallel indexing
        dataset = dataset_cls(
            parquet_files=data_paths,
            tokenizer=tokenizer,
            max_length=data_config.get('max_length', 1024),
            truncation=data_config.get('truncation', 'error'),
            num_workers=num_workers  # Use auto-detected number of workers
        )
    elif dataset_cls == MultiTurnSFTDataset:
        # MultiTurnSFTDataset uses config
        dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config)
    else:
        # Custom dataset classes might accept config
        try:
            dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config)
        except:
            # Fall back to SFTDataset interface
            dataset = dataset_cls(
                parquet_files=data_paths,
                tokenizer=tokenizer,
                max_length=data_config.get('max_length', 1024),
                truncation=data_config.get('truncation', 'error')
            )
    
    print(f"[DEBUG] Dataset created successfully, length: {len(dataset)}")
    return dataset


if __name__ == "__main__":
    main()
