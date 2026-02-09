---
name: flash-attention
description: Flash Attention, SDPA backends, PagedAttention, and attention kernel selection/configuration. Use when choosing or configuring attention backends for training or inference (FlashAttention-2/3, SDPA, xFormers, PagedAttention, Ring Attention).
---

# Flash Attention & Attention Backends

## Attention Backend Landscape

| Backend | GPU Arch | Dtypes | Max Head Dim | Use Case |
|---------|----------|--------|-------------|----------|
| FlashAttention-2 | Ampere, Ada, Hopper (CUDA); MI200/MI300 (ROCm) | fp16, bf16 | 256 | General training & inference |
| FlashAttention-3 | Hopper only (H100/H800) | fp16, bf16, fp8 (fwd only) | 256 | Maximum throughput on H100 |
| SDPA Math | Any CUDA | fp32, fp16, bf16 | Any | Fallback / debugging |
| SDPA Efficient (xFormers/Memory-Efficient) | Ampere+ | fp16, bf16 | 128 | When FA unavailable |
| PagedAttention | Ampere+ | fp16, bf16 | 128+ | KV cache management in inference (vLLM, TGI) |
| Ring Attention | Multi-GPU | fp16, bf16 | 256 | Sequence parallelism for very long contexts |
| FlexAttention | Ampere+ | fp16, bf16 | Any | Custom attention masks via `torch.nn.attention.flex_attention` |

## PyTorch SDPA (Scaled Dot Product Attention)

PyTorch routes `F.scaled_dot_product_attention()` to the best available backend automatically. Override with the context manager:

```python
from torch.nn.attention import SDPBackend, sdpa_kernel

# Force FlashAttention only
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    output = F.scaled_dot_product_attention(q, k, v)

# Set priority order (try Flash first, fall back to Efficient)
with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION], set_priority=True):
    output = F.scaled_dot_product_attention(q, k, v)
```

### SDPBackend Enum

| Backend | Enum Value | Notes |
|---------|-----------|-------|
| `FLASH_ATTENTION` | Flash Attention (Dao-AILab) | Requires Ampere+, fp16/bf16, head_dim ≤ 256 |
| `EFFICIENT_ATTENTION` | Memory-Efficient (xFormers-style) | Broader head_dim support, slightly slower |
| `MATH` | Naive PyTorch math | No restrictions, no fusion, slow — use for debugging |
| `CUDNN_ATTENTION` | cuDNN backend | Available in PyTorch 2.2+, limited config |

### Backend Selection Logic

PyTorch checks backends in priority order. A backend is **skipped** if:
- GPU arch not supported (e.g., FA on Turing T4)
- Dtype mismatch (e.g., fp32 → no FA)
- Head dimension exceeds backend limit
- Causal mask + custom attention mask combination unsupported
- Dropout > 0.0 with some backends in eval mode

Check which backend would be selected:

```python
from torch.nn.attention import _get_flash_version
# Returns version string or None if unavailable
print(_get_flash_version())
```

## FlashAttention-2 Direct API

The `flash-attn` package provides a direct API bypassing SDPA:

```python
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func

# Separate Q, K, V tensors
# q, k, v: (batch, seqlen, nheads, headdim)
output = flash_attn_func(
    q, k, v,
    dropout_p=0.0,          # Set 0.0 during eval
    softmax_scale=None,      # Default: 1/sqrt(headdim)
    causal=False,
    window_size=(-1, -1),    # Sliding window: (left, right)
    alibi_slopes=None,       # (nheads,) or (batch, nheads) for ALiBi
    deterministic=False,     # Deterministic backward (slower, more memory)
)

# Packed QKV (faster backward — avoids gradient concat)
# qkv: (batch, seqlen, 3, nheads, headdim)
output = flash_attn_qkvpacked_func(qkv, causal=True)
```

### Key Parameters

| Parameter | Type | Default | Effect |
|-----------|------|---------|--------|
| `dropout_p` | float | 0.0 | Attention dropout probability |
| `softmax_scale` | float | `1/sqrt(d)` | QK scaling factor |
| `causal` | bool | False | Apply causal mask |
| `window_size` | (int, int) | (-1, -1) | Sliding window attention bounds |
| `alibi_slopes` | Tensor | None | ALiBi positional bias slopes |
| `deterministic` | bool | False | Deterministic backward pass |

### GQA/MQA Support

Flash Attention supports grouped-query and multi-query attention natively. Pass K, V with fewer heads than Q — the number of Q heads must be divisible by K/V heads:

```python
# GQA: 32 query heads, 8 KV heads
q = torch.randn(B, S, 32, D, dtype=torch.float16, device="cuda")
k = torch.randn(B, S, 8, D, dtype=torch.float16, device="cuda")
v = torch.randn(B, S, 8, D, dtype=torch.float16, device="cuda")
output = flash_attn_func(q, k, v, causal=True)
```

## FlashAttention-3 (Hopper)

FA3 is optimized for H100/H800 with warp specialization and FP8 support:

```python
import flash_attn_interface

output = flash_attn_interface.flash_attn_func(
    q, k, v,
    causal=True,
    # Same API as FA2 but with FP8 forward support
)
```

**Requirements**: H100/H800, CUDA ≥ 12.3 (12.8 recommended for best performance).

**Key differences from FA2**:
- Uses asynchronous warp-specialized kernels
- FP8 forward pass (E4M3 format)
- ~1.5-2x faster than FA2 on H100 for fp16

## PagedAttention

PagedAttention manages KV cache memory in blocks (pages) rather than contiguous tensors — eliminates memory fragmentation during inference. Used by vLLM, TGI, and SGLang internally.

**Not directly configured by users** — it's an implementation detail of inference engines. However, understand its impact:

| Setting | Where | Effect |
|---------|-------|--------|
| `block_size` | vLLM `EngineArgs` | Page size for KV cache blocks (default: 16) |
| `gpu_memory_utilization` | vLLM | Fraction of GPU memory for KV cache (default: 0.9) |
| `swap_space` | vLLM | CPU swap space in GB for overflow pages |
| `num_gpu_blocks` | Computed | Total KV cache blocks available |

**How it works**: Instead of pre-allocating max_seq_len × batch_size contiguous memory, PagedAttention allocates fixed-size blocks on demand. Sequences reference block tables mapping logical positions to physical blocks. This allows:
- Near-zero memory waste from padding
- Efficient prefix caching (shared blocks across sequences)
- Copy-on-write for parallel sampling

## Ring Attention (Sequence Parallelism)

Ring Attention distributes long sequences across GPUs in a ring topology. Each GPU holds a chunk of Q and rotates K, V blocks around the ring:

- Enables training on sequences longer than single-GPU memory allows
- Linear memory scaling with number of GPUs
- Used in: Llama 3.1 (128K context), DeepSeek-V2

**Integration**: Typically activated via framework flags rather than direct API:
- **Megatron-LM**: `--context-parallel-size N` enables ring attention
- **DeepSpeed Ulysses**: Alternative approach — all-to-all on attention heads instead of ring on sequence

## FlexAttention (PyTorch 2.5+)

`flex_attention` compiles custom attention patterns into fused kernels:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# Define custom score modification
def causal_with_window(score, b, h, q_idx, kv_idx):
    # Causal + sliding window of 1024
    return torch.where(
        (q_idx >= kv_idx) & (q_idx - kv_idx < 1024),
        score,
        float("-inf"),
    )

# Create block mask for sparsity optimization
block_mask = create_block_mask(causal_with_window, B, H, Q_LEN, KV_LEN)

# Compiled attention with custom mask
output = flex_attention(q, k, v, block_mask=block_mask)
```

**Advantages**: Fuses arbitrary attention patterns (document masking, sliding window + causal, prefix LM) without materializing the full attention matrix.

## Hugging Face Transformers Integration

Control attention backend via `attn_implementation`:

```python
from transformers import AutoModelForCausalLM

# Explicit backend selection
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    attn_implementation="flash_attention_2",  # or "sdpa", "eager"
    torch_dtype=torch.bfloat16,
)
```

| Value | Backend | Notes |
|-------|---------|-------|
| `"flash_attention_2"` | FlashAttention-2 via `flash-attn` package | Requires `pip install flash-attn` in image |
| `"sdpa"` | PyTorch native SDPA | Default for PyTorch ≥ 2.1.1 |
| `"eager"` | Manual attention math | Debugging only, very slow |

## Container Image Considerations

FlashAttention compilation is slow (~30 min). Use pre-built wheels or NGC containers:

```dockerfile
# Option 1: NGC PyTorch container (includes FA)
FROM nvcr.io/nvidia/pytorch:24.12-py3

# Option 2: Pre-built wheel
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
RUN pip install flash-attn --no-build-isolation

# Option 3: Set MAX_JOBS to limit compilation parallelism
ENV MAX_JOBS=4
RUN pip install flash-attn --no-build-isolation
```

## ROCm Support

FlashAttention-2 on AMD GPUs (MI200, MI250, MI300) supports two backends:

| Backend | Install | GPUs | Notes |
|---------|---------|------|-------|
| Composable Kernel (CK) | Default | MI200/MI250/MI300/MI355 | Head dim ≤ 256 |
| Triton | `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` | MI200/MI300 + RDNA | Adds FP8, paged attention, rotary embeddings |

Triton backend env vars for tuning:

| Variable | Effect |
|----------|--------|
| `FLASH_ATTENTION_TRITON_AMD_ENABLE` | Enable Triton backend |
| `FLASH_ATTENTION_TRITON_AMD_AUTOTUNE` | Enable autotuning (slower first run) |
| `FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON` | Override tile config: `{"BLOCK_M":128,"BLOCK_N":64}` |

## Troubleshooting

### "Torch was not compiled with flash attention"

This warning means the PyTorch build lacks FA support. Causes:
1. GPU arch < Ampere (T4, V100) — FA2 requires SM80+
2. CUDA toolkit version mismatch
3. PyTorch built without FA kernel

**Fix**: Install `flash-attn` package separately, or use `attn_implementation="sdpa"` which uses the Efficient Attention fallback.

### Head Dimension Errors

FA2 supports head_dim ≤ 256, but backward pass for head_dim > 192 requires A100/H100. If training with head_dim 256 on consumer GPUs, disable dropout (`dropout_p=0.0`).

### Memory Errors with Long Sequences

Flash Attention is O(N) memory in sequence length (vs O(N²) for vanilla attention), but KV cache still grows linearly. For very long sequences:
- Use sliding window attention (`window_size` parameter)
- Enable Ring Attention for multi-GPU sequence parallelism
- Consider chunked prefill in inference engines
