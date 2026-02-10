---
name: pytorch
description: >
  Write deep learning code with PyTorch — models, training loops, distributed training, and
  performance optimization. Use when: (1) Building nn.Module architectures (layers, custom
  modules, transformers), (2) Writing training loops (DataLoader, optimizers, loss functions),
  (3) Implementing distributed training (DDP, FSDP, torchrun), (4) Using mixed precision
  (torch.amp, GradScaler, bf16/fp16), (5) Optimizing performance (torch.compile, profiler,
  memory), (6) Serializing models (state_dict, TorchScript, torch.export, ONNX), (7) Using
  modern idioms (scaled_dot_product_attention, flex_attention), (8) Debugging CUDA OOM, NaN
  gradients, or NCCL hangs.
---

# PyTorch

PyTorch is the Python deep learning framework. Version: **2.5+**.

## Building Models

### nn.Module Basics

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_features: int, hidden: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.gelu(self.norm(self.fc1(x))))
        return self.fc2(x)
```

### Common Layers

| Layer | Purpose | Example |
|-------|---------|---------|
| `nn.Linear(in, out)` | Dense/fully connected | `nn.Linear(768, 256)` |
| `nn.Conv2d(in, out, k)` | 2D convolution | `nn.Conv2d(3, 64, 3, padding=1)` |
| `nn.TransformerEncoderLayer` | Transformer block | `nn.TransformerEncoderLayer(d_model=512, nhead=8)` |
| `nn.Embedding(num, dim)` | Lookup table | `nn.Embedding(50257, 768)` |
| `nn.LSTM(in, hidden)` | LSTM recurrent | `nn.LSTM(256, 512, num_layers=2, batch_first=True)` |
| `nn.LayerNorm(dim)` | Layer normalization | `nn.LayerNorm(768)` |
| `nn.BatchNorm2d(channels)` | Batch normalization | `nn.BatchNorm2d(64)` |
| `nn.MultiheadAttention` | Multi-head attention | `nn.MultiheadAttention(768, 12)` |
| `nn.ModuleList` | List of submodules | `nn.ModuleList([Block() for _ in range(12)])` |
| `nn.Sequential` | Sequential container | `nn.Sequential(nn.Linear(768, 256), nn.ReLU())` |

### Custom Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-norm architecture
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, attn_mask=mask, need_weights=False)[0]
        x = x + self.ff(self.norm2(x))
        return x
```

### Modern Attention (PyTorch 2.0+)

```python
# scaled_dot_product_attention — fused, memory-efficient, Flash Attention
from torch.nn.functional import scaled_dot_product_attention

class EfficientAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # Automatically selects Flash Attention / memory-efficient backend
        out = scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).reshape(B, T, C))
```

## Training Loop

### Standard Training Loop

```python
import torch
from torch.utils.data import DataLoader, Dataset

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-4,
    device: str = "cuda",
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                correct += (outputs.argmax(1) == targets).sum().item()
                total += targets.size(0)

        print(f"Epoch {epoch}: loss={total_loss/len(train_loader):.4f}, "
              f"val_acc={correct/total:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
```

### DataLoader Best Practices

```python
train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=8,          # parallel data loading
    pin_memory=True,        # faster CPU→GPU transfer
    persistent_workers=True, # keep workers alive between epochs
    prefetch_factor=2,      # batches to prefetch per worker
    drop_last=True,         # consistent batch sizes for DDP
)
```

### Optimizers

```python
# AdamW — default for most tasks
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# SGD with momentum — vision models
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

# Per-parameter options (e.g., no weight decay on bias/norm)
param_groups = [
    {"params": [p for n, p in model.named_parameters() if "bias" not in n and "norm" not in n],
     "weight_decay": 0.01},
    {"params": [p for n, p in model.named_parameters() if "bias" in n or "norm" in n],
     "weight_decay": 0.0},
]
optimizer = torch.optim.AdamW(param_groups, lr=1e-4)
```

### Learning Rate Schedulers

```python
# Cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Warmup + cosine (common for transformers)
from torch.optim.lr_scheduler import LambdaLR

def warmup_cosine(step, warmup_steps=1000, total_steps=10000):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine)

# OneCycleLR — good default for quick experiments
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, total_steps=len(train_loader) * epochs
)
```

## Mixed Precision Training

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler("cuda")

for batch in train_loader:
    inputs, targets = batch[0].to(device), batch[1].to(device)
    optimizer.zero_grad(set_to_none=True)

    with autocast("cuda", dtype=torch.bfloat16):  # or torch.float16
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

**bf16 vs fp16:**
| | bf16 | fp16 |
|--|------|------|
| Range | Same as fp32 | Narrower (needs GradScaler) |
| Precision | Lower mantissa | Higher mantissa |
| Hardware | A100, H100, 4090 | All NVIDIA GPUs |
| GradScaler needed | No (optional) | Yes |
| Recommendation | **Prefer bf16 if available** | Use on older GPUs (V100, T4) |

## Distributed Training

### DDP (DistributedDataParallel)

The standard for multi-GPU training. Each process gets a full model copy; gradients are synchronized.

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def train_ddp(local_rank: int, world_size: int):
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)

    model = MyModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler, pin_memory=True)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # critical for proper shuffling
        for batch in loader:
            # same training loop as single GPU
            ...

    dist.destroy_process_group()
```

**Launch with torchrun:**
```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 train.py

# Multi-node (run on each node)
torchrun --nproc_per_node=4 --nnodes=2 \
  --node_rank=0 --master_addr=10.0.0.1 --master_port=29500 \
  train.py
```

### FSDP (Fully Sharded Data Parallel)

For models that don't fit on a single GPU — shards parameters, gradients, and optimizer state:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

# Auto-wrap policy: shard per transformer block
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock},
)

mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=mp_policy,
    device_id=local_rank,
    use_orig_params=True,  # required for torch.compile
)
```

## torch.compile (PyTorch 2.0+)

Compiles the model into optimized kernels:

```python
model = torch.compile(model)  # default mode

# Mode options
model = torch.compile(model, mode="reduce-overhead")  # best for small models, uses CUDA graphs
model = torch.compile(model, mode="max-autotune")     # slower compile, fastest runtime
model = torch.compile(model, fullgraph=True)           # error if graph breaks

# Compile only the training step
@torch.compile
def train_step(model, inputs, targets, criterion):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    return loss
```

**Debug graph breaks:**
```python
import torch._dynamo as dynamo
dynamo.config.verbose = True
# or
torch._dynamo.config.suppress_errors = False
```

## Model Serialization

### state_dict (Recommended)

```python
# Save
torch.save(model.state_dict(), "model.pt")

# Load
model = MyModel()
model.load_state_dict(torch.load("model.pt", map_location="cpu", weights_only=True))
```

### torch.export (PyTorch 2.1+)

```python
# Export for deployment (captures full graph)
exported = torch.export.export(model, (example_input,))
torch.export.save(exported, "model.pt2")

# Load
loaded = torch.export.load("model.pt2")
output = loaded.module()(input_tensor)
```

### ONNX Export

```python
torch.onnx.export(
    model,
    example_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
)
```

## Performance Optimization

### Memory Management

```python
# Empty cache (for debugging, not in training loops)
torch.cuda.empty_cache()

# Monitor memory
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Memory snapshot for debugging
torch.cuda.memory._record_memory_history()
# ... run your code ...
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
# Visualize at https://pytorch.org/memory_viz

# Gradient checkpointing — trade compute for memory
from torch.utils.checkpoint import checkpoint
class Block(nn.Module):
    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)
    def _forward(self, x):
        ...
```

### Profiler

```python
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler("./profiler_logs"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(train_loader):
        train_step(model, batch)
        prof.step()
        if step >= 6:
            break

# View: tensorboard --logdir=./profiler_logs
```

## torch.export (Model Export)

Export models for deployment:

```python
import torch

model = MyModel()
example_input = torch.randn(1, 3, 224, 224)

# Export (graph capture, no Python overhead)
exported = torch.export.export(model, (example_input,))

# Save
torch.export.save(exported, "model.pt2")

# Load
loaded = torch.export.load("model.pt2")
output = loaded.module()(example_input)

# Export to ONNX
torch.onnx.export(model, example_input, "model.onnx", opset_version=17)
```

## Custom Autograd Functions

```python
class CustomMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a @ b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return grad_output @ b.T, a.T @ grad_output

# Use: result = CustomMatMul.apply(tensor_a, tensor_b)
```

## Debugging

See `references/troubleshooting.md` for:
- CUDA out of memory
- NaN/Inf gradients
- Slow DataLoader
- NCCL distributed training hangs
- torch.compile graph breaks
- Common numerical issues

## Cross-References

- [fsdp](../fsdp/) — FSDP for memory-efficient distributed training
- [megatron-lm](../megatron-lm/) — Megatron-LM for large-scale training
- [ray-train](../ray-train/) — Ray Train for managed distributed training
- [flash-attention](../flash-attention/) — Attention backend selection (FA2, FA3, SDPA)
- [torch-compile](../torch-compile/) — torch.compile for training and inference optimization

## Reference

- [PyTorch docs](https://pytorch.org/docs/stable/)
- [PyTorch tutorials](https://pytorch.org/tutorials/)
- [torch.compile tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [FSDP tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- `references/troubleshooting.md` — common errors and fixes
- `scripts/benchmark_matmul.py` — GPU matmul throughput benchmark across dtypes and sizes
