---
name: fsdp
description: >
  Distributed training with PyTorch FSDP (Fully Sharded Data Parallel) — shard model parameters,
  gradients, and optimizer state across GPUs. Use when: (1) Training models too large for a single
  GPU, (2) Configuring sharding strategies (FULL_SHARD, SHARD_GRAD_OP, NO_SHARD), (3) Setting up
  mixed precision with FSDP, (4) Implementing activation checkpointing, (5) Wrapping transformer
  layers with auto_wrap_policy, (6) Saving and loading FSDP checkpoints, (7) Using FSDP with
  HuggingFace Trainer or Accelerate, (8) Debugging FSDP-specific issues (hangs, OOM, checkpoint
  problems).
---

# FSDP (Fully Sharded Data Parallel)

PyTorch FSDP shards model parameters, gradients, and optimizer states across GPUs — enabling training of models that don't fit on a single GPU. Part of `torch.distributed.fsdp`. PyTorch **2.5+**.

> **When to use FSDP vs DDP**: Use DDP when the model fits on one GPU. Use FSDP when it doesn't (typically >10B parameters, or large batch sizes exceeding single-GPU memory).

## Core Concepts

FSDP shards a model's parameters across `N` GPUs. During forward/backward:
1. **All-gather** parameters for the current layer (briefly full on each GPU)
2. Compute forward/backward
3. **Reduce-scatter** gradients
4. Discard non-local shards

This trades communication for memory — each GPU only stores `1/N` of parameters + optimizer state.

## Sharding Strategies

| Strategy | Memory Savings | Communication | Use Case |
|----------|---------------|---------------|----------|
| `FULL_SHARD` | Maximum (params + grads + optimizer) | Highest | Default — models that don't fit |
| `SHARD_GRAD_OP` | Moderate (grads + optimizer only) | Lower | Model fits but optimizer doesn't |
| `NO_SHARD` | None (equivalent to DDP) | Lowest | Baseline / debugging |
| `HYBRID_SHARD` | Full shard within node, replicate across | Balanced | Multi-node with fast intra-node links |

## Basic Setup

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
import functools

def train(local_rank: int):
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)

    model = build_model()

    # Auto-wrap policy: each transformer layer becomes an FSDP unit
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},  # your layer class
    )

    # Mixed precision
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=local_rank,
        use_orig_params=True,       # required for torch.compile
        limit_all_gathers=True,     # limit memory from all-gathers
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch).loss
            loss.backward()
            model.clip_grad_norm_(1.0)  # FSDP-aware grad clipping
            optimizer.step()

    dist.destroy_process_group()
```

**Launch:**
```bash
torchrun --nproc_per_node=4 train.py
# Multi-node:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
  --master_addr=10.0.0.1 --master_port=29500 train.py
```

## Wrap Policies

### Transformer Auto-Wrap

Wraps each transformer layer as a separate FSDP unit — the standard for LLMs:

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# For HuggingFace models, import the layer class:
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer},
)
```

### Size-Based Wrap

Wraps modules exceeding a parameter threshold:

```python
auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy,
    min_num_params=1_000_000,  # 1M params
)
```

### Custom Wrap Policy

```python
from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy

def custom_policy(module, recurse, **kwargs):
    if recurse:
        return True  # always recurse into children
    # Wrap specific module types
    return isinstance(module, (TransformerBlock, nn.Embedding))

auto_wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=custom_policy)
```

## Mixed Precision

```python
# bf16 — recommended for A100/H100
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,    # parameters stored in bf16
    reduce_dtype=torch.bfloat16,   # gradient reduction in bf16
    buffer_dtype=torch.bfloat16,   # buffers (e.g., BatchNorm) in bf16
)

# fp16 — for older GPUs (V100/T4), needs loss scaling
mp_policy = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
)

# Keep some ops in fp32 for stability (e.g., loss computation)
# FSDP handles this via param_dtype — the forward pass upcasts as needed
```

## Activation Checkpointing

Trade compute for memory — recompute activations during backward instead of storing them:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.checkpoint import checkpoint

# Option 1: Apply to the FSDP-wrapped model
from torch.distributed.fsdp import apply_activation_checkpointing
import functools

apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=functools.partial(
        checkpoint_wrapper,
        checkpoint_fn=checkpoint,
    ),
    check_fn=lambda submodule: isinstance(submodule, TransformerBlock),
)

# Option 2: With HuggingFace — just enable in TrainingArguments
# gradient_checkpointing=True (see HF integration below)
```

## Checkpointing

### Full State Dict (For Inference / Single-GPU Loading)

```python
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

# Save
save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
    state_dict = model.state_dict()
    if dist.get_rank() == 0:
        torch.save(state_dict, "model.pt")

# Load (on any device)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
```

### Sharded State Dict (For Resuming Training)

Faster save/load — each rank saves its own shard:

```python
from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType
from torch.distributed.checkpoint import save, load

# Save
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    save(state_dict, checkpoint_id="checkpoint-epoch-1")

# Load
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    load(state_dict, checkpoint_id="checkpoint-epoch-1")
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
```

## CPU Offloading

Offload parameters and gradients to CPU when not in use:

```python
model = FSDP(
    model,
    cpu_offload=CPUOffload(offload_params=True),
    # Note: significantly slower but enables very large models on limited GPUs
)
```

## HuggingFace Integration

### With Trainer + Accelerate

The easiest way to use FSDP with HuggingFace models:

```yaml
# fsdp_config.yaml (accelerate config)
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_offload_params: false
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_use_orig_params: true
mixed_precision: bf16
num_machines: 1
num_processes: 4
```

```python
training_args = TrainingArguments(
    output_dir="./results",
    fsdp="full_shard auto_wrap",
    fsdp_config="fsdp_config.yaml",
    bf16=True,
    gradient_checkpointing=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    ...
)
```

```bash
accelerate launch --config_file fsdp_config.yaml train.py
# Or directly with torchrun (Trainer auto-detects FSDP from args)
torchrun --nproc_per_node=4 train.py
```

### With Accelerate (Manual)

```python
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

fsdp_plugin = FullyShardedDataParallelPlugin(
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision_policy=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    ),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    loss = model(**batch).loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

## FSDP2 (torch.distributed._composable.fsdp)

PyTorch 2.4+ introduces FSDP2 — a composable, per-parameter sharding API:

```python
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)

# Apply per-module (more granular control)
for layer in model.layers:
    fully_shard(layer, mp_policy=mp_policy)
fully_shard(model, mp_policy=mp_policy)
```

**Key differences from FSDP1:**
- Per-parameter sharding (composable with TP, CP, etc.)
- No wrapper — modifies model in-place
- Better integration with `torch.compile`
- Used by TorchTitan and Megatron-LM Bridge

## Tensor Parallel + FSDP (2D Parallelism)

Combine FSDP (data parallel) with TP (model parallel) for very large models:

```python
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed._composable.fsdp import fully_shard

# 1. Apply tensor parallelism within each node
parallelize_module(model, tp_mesh, {
    "attention.q_proj": ColwiseParallel(),
    "attention.v_proj": ColwiseParallel(),
    "attention.o_proj": RowwiseParallel(),
    "mlp.gate_proj": ColwiseParallel(),
    "mlp.down_proj": RowwiseParallel(),
})

# 2. Apply FSDP across nodes
for layer in model.layers:
    fully_shard(layer, mesh=dp_mesh)
fully_shard(model, mesh=dp_mesh)
```

## torch.compile with FSDP

```python
# use_orig_params=True is required
model = FSDP(model, use_orig_params=True, ...)

# Compile after wrapping
model = torch.compile(model)
```

## HYBRID_SHARD (Multi-Node)

Full shard within a node, replicate across nodes — reduces inter-node communication:

```python
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
    auto_wrap_policy=auto_wrap_policy,
    device_id=local_rank,
)
```

Best when: intra-node bandwidth >> inter-node bandwidth (e.g., NVLink within, ethernet across).

## Debugging

See `references/troubleshooting.md` for:
- FSDP hangs and deadlocks
- OOM despite sharding
- Checkpoint save/load issues
- Mixed precision instability
- torch.compile incompatibilities

## Cross-References

- [pytorch](../pytorch/) — PyTorch distributed training fundamentals
- [ray-train](../ray-train/) — FSDP integration with Ray Train
- [megatron-lm](../megatron-lm/) — Alternative: Megatron-LM for very large models

## Reference

- [FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [FSDP Advanced Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_adv_tutorial.html)
- [FSDP API Reference](https://pytorch.org/docs/stable/fsdp.html)
- [Accelerate FSDP docs](https://huggingface.co/docs/accelerate/usage_guides/fsdp)
- `references/troubleshooting.md` — common errors and fixes
