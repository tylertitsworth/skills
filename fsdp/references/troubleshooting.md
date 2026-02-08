# FSDP Troubleshooting

## Hangs and Deadlocks

### All ranks not reaching a collective

FSDP uses collectives (all-gather, reduce-scatter) that must be called by all ranks.

**Common cause**: Conditional logic that differs across ranks:
```python
# Wrong — only some ranks may enter this branch
if batch_has_data:
    loss = model(batch)  # triggers all-gather — hangs if other ranks skip this

# Right — all ranks must call forward, even with empty batches
loss = model(batch)  # ensure all ranks get a batch (use drop_last=True)
```

**Debug:**
```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=300
```

### Hang during initialization

```python
# Ensure all ranks see the same model before wrapping
model = MyModel()
# Don't put model on GPU before FSDP wrapping
model = FSDP(model, device_id=local_rank)  # FSDP handles device placement
```

## OOM Despite Sharding

### Model still doesn't fit

1. **Check wrap policy is working**: Each FSDP unit is sharded independently. If the whole model is one unit, no memory savings:
   ```python
   # Verify wrapping
   print(model)  # should show nested FSDP units
   ```

2. **Enable activation checkpointing** — activations often dominate memory:
   ```python
   apply_activation_checkpointing(model, check_fn=lambda m: isinstance(m, TransformerBlock))
   ```

3. **Enable CPU offloading** (slow but saves GPU memory):
   ```python
   model = FSDP(model, cpu_offload=CPUOffload(offload_params=True))
   ```

4. **Reduce batch size** — each GPU processes its local batch

5. **Use `limit_all_gathers=True`** — prevents multiple layers from all-gathering simultaneously:
   ```python
   model = FSDP(model, limit_all_gathers=True)
   ```

### OOM during all-gather

During forward/backward, FSDP briefly materializes full parameters for one FSDP unit. If a single unit is too large:
- Wrap at a finer granularity (e.g., individual attention/FF modules instead of entire blocks)
- Use `size_based_auto_wrap_policy` with a lower threshold

## Checkpoint Issues

### "RuntimeError: ShardedTensor metadata mismatch"

Checkpoint was saved with different number of ranks than loading. Solutions:
- Use `FULL_STATE_DICT` for portable checkpoints
- Save with `offload_to_cpu=True, rank0_only=True` for single-file checkpoints

### Checkpoint too slow

`FULL_STATE_DICT` gathers everything to rank 0 — slow for large models.

**Fix**: Use `SHARDED_STATE_DICT` with `torch.distributed.checkpoint`:
```python
from torch.distributed.checkpoint import save, load

with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    save({"model": model.state_dict()}, checkpoint_id="ckpt")
```

### Loading checkpoint into non-FSDP model

Save as `FULL_STATE_DICT` first:
```python
save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
    state = model.state_dict()
    if dist.get_rank() == 0:
        torch.save(state, "model_full.pt")

# Load on single GPU
model = MyModel()
model.load_state_dict(torch.load("model_full.pt", map_location="cpu"))
```

## Mixed Precision Issues

### NaN loss with fp16

fp16 has narrow range — gradients can overflow:
```python
# Switch to bf16 (wider range, no loss scaling needed)
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
)
```

### Accuracy degradation with bf16

Some operations need fp32 precision (e.g., loss computation, softmax):
```python
# Keep buffers in fp32
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,  # reduce in fp32 for better accuracy
    buffer_dtype=torch.float32,
)
```

## torch.compile Issues

### "RuntimeError: use_orig_params=True is required"

```python
model = FSDP(model, use_orig_params=True, ...)
model = torch.compile(model)
```

### Graph breaks with FSDP

FSDP's parameter gathering can cause graph breaks:
```python
# Try compiling individual modules before FSDP wrapping
model.transformer.compile()  # compile sub-modules
model = FSDP(model, use_orig_params=True)
```

## HuggingFace Integration Issues

### Wrong transformer layer class

```
ValueError: Could not find the transformer layer class to wrap
```

Find the correct class:
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_name)
for name, module in model.named_modules():
    print(name, type(module).__name__)
# Look for "DecoderLayer", "Block", "TransformerLayer", etc.
```

Common mappings:
| Model | Layer Class |
|-------|------------|
| Llama | `LlamaDecoderLayer` |
| Mistral | `MistralDecoderLayer` |
| GPT-2 | `GPT2Block` |
| Falcon | `FalconDecoderLayer` |
| Phi | `PhiDecoderLayer` |
| Gemma | `GemmaDecoderLayer` |
| Qwen2 | `Qwen2DecoderLayer` |

### Accelerate config issues

Run the setup wizard:
```bash
accelerate config  # interactive
# Or write the YAML directly (see SKILL.md for template)
```
