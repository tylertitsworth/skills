# PyTorch Troubleshooting

## CUDA Out of Memory

### Diagnosis

```python
# Check current allocation
print(torch.cuda.memory_summary())

# Find the largest tensors
import gc
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) and obj.is_cuda:
            print(type(obj), obj.size(), obj.dtype, obj.device)
    except:
        pass
```

### Fixes (ordered by impact)

1. **Reduce batch size** — most common fix
2. **Enable gradient checkpointing:**
   ```python
   model.gradient_checkpointing_enable()  # HuggingFace models
   # Or manually with torch.utils.checkpoint
   ```
3. **Use mixed precision** — halves activation memory:
   ```python
   with torch.amp.autocast("cuda", dtype=torch.bfloat16):
       ...
   ```
4. **Clear unused tensors:**
   ```python
   del intermediate_tensor
   torch.cuda.empty_cache()
   ```
5. **Accumulate gradients** instead of larger batches:
   ```python
   accumulation_steps = 4
   for i, batch in enumerate(loader):
       loss = model(batch) / accumulation_steps
       loss.backward()
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad(set_to_none=True)
   ```
6. **Use FSDP** to shard model across GPUs
7. **Set `max_split_size_mb`** to reduce fragmentation:
   ```bash
   PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
   ```

## NaN / Inf Gradients

### Detection

```python
# Check after backward
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            print(f"NaN gradient in {name}")
        if torch.isinf(param.grad).any():
            print(f"Inf gradient in {name}")

# Enable anomaly detection (slow, use for debugging only)
torch.autograd.set_detect_anomaly(True)
```

### Common Causes and Fixes

**Learning rate too high:**
```python
# Reduce LR or use warmup
scheduler = LambdaLR(optimizer, lambda step: min(1.0, step / 1000))
```

**Loss function issues:**
```python
# log(0) = -inf — add epsilon or use built-in functions
loss = F.cross_entropy(logits, targets)  # numerically stable
# NOT: loss = -torch.log(softmax_output[targets])
```

**Division by zero in normalization:**
```python
# Add epsilon
normalized = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
```

**fp16 overflow:**
```python
# Switch to bf16 (wider range) or ensure GradScaler is used
with torch.amp.autocast("cuda", dtype=torch.bfloat16):  # prefer bf16
    ...
```

**Gradient clipping:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# Or clip by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
```

## Slow DataLoader

### Symptoms
Training is GPU-bound but GPU utilization is low (data loading is the bottleneck).

### Fixes

```python
loader = DataLoader(
    dataset,
    num_workers=8,           # increase workers (rule of thumb: 4 per GPU)
    pin_memory=True,         # pre-allocate in pinned (page-locked) memory
    persistent_workers=True,  # don't restart workers each epoch
    prefetch_factor=2,       # prefetch 2 batches per worker
)
```

**Move preprocessing to GPU:**
```python
# Instead of CPU transforms in dataset.__getitem__:
# Do transforms on GPU in the training loop
batch = batch.to(device, non_blocking=True)  # async transfer with pin_memory
batch = gpu_transform(batch)
```

**Use IterableDataset for streaming:**
```python
class StreamingDataset(torch.utils.data.IterableDataset):
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # split work across workers
        for sample in self.shard(worker_info):
            yield process(sample)
```

**Pre-cache to fast storage:**
```bash
# Copy dataset to NVMe SSD or tmpfs before training
cp -r /nfs/dataset /local-ssd/dataset
```

## NCCL Distributed Training Hangs

### Diagnosis

```bash
# Enable NCCL debug logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Set a timeout so hangs become errors
export NCCL_TIMEOUT=300  # seconds
```

```python
# In code
dist.init_process_group("nccl", timeout=timedelta(minutes=5))
```

### Common Causes

**Not all ranks reach a collective:**
```python
# Wrong — only rank 0 calls all_reduce
if rank == 0:
    dist.all_reduce(tensor)  # hangs — other ranks never call this

# Right — all ranks must call collectives
dist.all_reduce(tensor)  # called by all ranks
```

**Mismatched tensor shapes across ranks:**
```python
# Ensure all ranks use the same batch size (use drop_last=True)
loader = DataLoader(dataset, batch_size=64, drop_last=True, sampler=DistributedSampler(...))
```

**Network issues:**
```bash
# Test NCCL connectivity
NCCL_DEBUG=INFO torchrun --nproc_per_node=2 -c "import torch.distributed; torch.distributed.init_process_group('nccl')"

# Use specific network interface
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0

# Disable InfiniBand if not available
export NCCL_IB_DISABLE=1
```

**Firewall blocking ports:**
```bash
# torchrun uses port 29500 by default, plus ephemeral ports for NCCL
# Ensure ports are open between nodes
```

### Debugging with TORCH_DISTRIBUTED_DEBUG

```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# Logs collective operations, helps find which op hangs
```

## torch.compile Issues

### Graph breaks

Graph breaks split the compiled region, reducing performance.

```python
# Find graph breaks
import torch._dynamo as dynamo
dynamo.config.verbose = True

# Common causes:
# 1. Data-dependent control flow
if x.item() > 0:  # BREAKS — .item() materializes
    ...

# 2. Print statements
print(x.shape)  # BREAKS

# 3. Unsupported operations
# Check: torch._dynamo.list_backends()
```

**Fix**: Move dynamic logic outside compiled regions, or use `torch.cond`:
```python
@torch.compile
def forward(x):
    # Keep it tensor-only, no Python control flow on tensor values
    return F.relu(self.linear(x))
```

### "Skipping frame" warnings

```python
# Suppress for known-safe cases
import torch._dynamo
torch._dynamo.config.suppress_errors = True  # last resort

# Or mark functions as not-compilable
@torch._dynamo.disable
def non_compilable_fn():
    ...
```

### CUDA graph issues with reduce-overhead

```python
# reduce-overhead uses CUDA graphs, which require static shapes
# If shapes vary, use default mode instead
model = torch.compile(model, mode="default")
```

## Checkpoint Loading Issues

### "Missing key" / "Unexpected key"

```python
# strict=False ignores mismatched keys
model.load_state_dict(torch.load("ckpt.pt", weights_only=True), strict=False)

# For DDP checkpoints (keys have "module." prefix)
state_dict = torch.load("ddp_ckpt.pt", weights_only=True)
new_state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
```

### "Attempting to deserialize on a CUDA device"

```python
# Load to CPU first, then move
state_dict = torch.load("model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)
model = model.to("cuda")
```

### weights_only=True errors

PyTorch 2.x defaults to `weights_only=True` for security. If you saved non-tensor objects:
```python
# If you trust the source
state_dict = torch.load("old_ckpt.pt", weights_only=False)

# Better: re-save with only state_dict
torch.save(model.state_dict(), "clean_ckpt.pt")
```

## Numerical Stability

### Softmax overflow

```python
# Wrong
probs = torch.exp(logits) / torch.exp(logits).sum(dim=-1, keepdim=True)

# Right — F.softmax subtracts the max internally
probs = F.softmax(logits, dim=-1)

# For log-softmax (used with NLLLoss)
log_probs = F.log_softmax(logits, dim=-1)
```

### Loss of precision in reductions

```python
# For fp16/bf16, accumulate in fp32
total = torch.tensor(0.0, dtype=torch.float32, device="cuda")
for x in tensor_list:
    total += x.float()
```
