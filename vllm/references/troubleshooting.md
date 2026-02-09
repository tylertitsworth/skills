# vLLM Troubleshooting

## GPU OOM (Out of Memory)

### During model loading

The model weights don't fit in GPU memory.

**Fixes (ordered by preference):**

1. **Enable quantization:**
   ```bash
   vllm serve my-model --quantization fp8    # 8-bit, minimal quality loss
   vllm serve my-model --quantization awq    # 4-bit (needs pre-quantized model)
   ```

2. **Use tensor parallelism** to shard across GPUs:
   ```bash
   vllm serve my-model --tensor-parallel-size 2
   ```

3. **Reduce max model length:**
   ```bash
   vllm serve my-model --max-model-len 4096  # reduces KV cache allocation
   ```

4. **Lower GPU memory utilization** (counterintuitively helps with fragmentation):
   ```bash
   vllm serve my-model --gpu-memory-utilization 0.85
   ```

### During inference

KV cache is full — too many concurrent sequences or long contexts.

**Fixes:**

1. **Reduce `--max-num-seqs`:**
   ```bash
   vllm serve my-model --max-num-seqs 64  # fewer concurrent requests
   ```

2. **Reduce `--max-model-len`:**
   ```bash
   vllm serve my-model --max-model-len 4096
   ```

3. **Enable prefix caching** (reuse KV cache for shared prefixes):
   ```bash
   vllm serve my-model --enable-prefix-caching
   ```

4. **Enable chunked prefill** (processes long prompts in chunks):
   ```bash
   vllm serve my-model --enable-chunked-prefill
   ```

### Memory estimation

```python
# Rough model memory (weights only):
# Parameters × bytes_per_param
# 7B model × 2 bytes (fp16) = ~14 GB
# 70B model × 2 bytes (fp16) = ~140 GB
# 70B model × 1 byte (fp8) = ~70 GB

# KV cache per token per layer (fp16):
# 2 × num_heads × head_dim × 2 bytes × num_layers
# Llama-3.1-8B: 2 × 32 × 128 × 2 × 32 = ~512 KB per token
```

## Slow Time-to-First-Token (TTFT)

TTFT is the time from request receipt to first generated token. Slow TTFT usually means the prefill (prompt processing) phase is bottlenecked.

### Diagnosis

Check the `/metrics` endpoint:
```bash
curl http://localhost:8000/metrics | grep time_to_first_token
```

### Fixes

1. **Enable chunked prefill** — allows decode steps during long prefills:
   ```bash
   vllm serve my-model --enable-chunked-prefill
   ```

2. **Reduce `--max-num-batched-tokens`** — limits prefill batch size:
   ```bash
   vllm serve my-model --max-num-batched-tokens 4096
   ```

3. **Enable prefix caching** — skips re-processing shared prefixes:
   ```bash
   vllm serve my-model --enable-prefix-caching
   ```

4. **Use FP8 quantization** — faster compute:
   ```bash
   vllm serve my-model --quantization fp8
   ```

5. **Use speculative decoding** — improves decode speed (but not prefill):
   ```bash
   vllm serve my-model --speculative-model small-model --num-speculative-tokens 5
   ```

## Throughput Degradation

### Symptoms
Throughput drops as concurrency increases.

### Diagnosis

```bash
# Check queue depth
curl http://localhost:8000/metrics | grep -E "num_requests_(running|waiting)"

# Check KV cache usage
curl http://localhost:8000/metrics | grep gpu_cache_usage
```

### Fixes

1. **KV cache full** (requests queuing): Increase `--gpu-memory-utilization` or reduce `--max-model-len`

2. **Batch too small** (GPU underutilized): Increase `--max-num-seqs`:
   ```bash
   vllm serve my-model --max-num-seqs 512
   ```

3. **Prefill-heavy workload** (long prompts dominate): Enable chunked prefill

4. **Scale out**: Add more replicas behind a load balancer:
   ```bash
   # Use round-robin or least-connections LB
   # Scale with HPA on vllm_num_requests_waiting metric
   ```

## Model Loading Failures

### "ValueError: The model's max seq len is too large"

The model's config specifies a context length larger than what fits in memory.

```bash
# Override with a smaller value
vllm serve my-model --max-model-len 8192
```

### "OSError: model not found" / Authentication errors

```bash
# Set HuggingFace token for gated models
export HUGGING_FACE_HUB_TOKEN="hf_..."
# Or
vllm serve my-model --download-dir /path/to/cache

# For offline/air-gapped:
# 1. Download model on a connected machine
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ./model
# 2. Serve from local path
vllm serve ./model
```

### "torch.cuda.OutOfMemoryError" during loading

Model weights exceed single GPU. Use tensor parallelism:
```bash
vllm serve large-model --tensor-parallel-size 2
```

Or quantize:
```bash
vllm serve large-model --quantization fp8
```

### Custom code models

Some models require `trust-remote-code`:
```bash
vllm serve my-custom-model --trust-remote-code
```

## Multi-GPU Communication Errors

### Worker crashed ("remote process exited")

One GPU worker crashed. Check:
1. GPU memory on all devices: `nvidia-smi`
2. GPU health: `nvidia-smi -q -d ECC`
3. Driver version matches across nodes
4. `/dev/shm` is large enough (mount as emptyDir with `medium: Memory` in K8s)

## CUDA Graph Issues

### "CUDA error: out of memory" during warmup

CUDA graphs pre-allocate memory during warmup. If memory is tight:

```bash
# Disable CUDA graphs (slower but less memory)
vllm serve my-model --enforce-eager

# Or reduce the number of captured batch sizes
vllm serve my-model --max-num-seqs 64  # fewer graph variants
```

### Crashes with dynamic shapes

CUDA graphs require static shapes. If you see graph-related crashes:
```bash
vllm serve my-model --enforce-eager  # fall back to eager mode
```

## LoRA Adapter Issues

### "LoRA adapter not found"

```bash
# Verify adapter path and enable LoRA
vllm serve base-model --enable-lora --lora-modules adapter1=/path/to/adapter
```

### Too many adapters / OOM

```bash
# Limit concurrent LoRA adapters in memory
--max-loras 4                   # max loaded simultaneously
--max-lora-rank 64              # max LoRA rank supported
--lora-extra-vocab-size 256     # extra vocab for adapter tokens
```

## Structured Output / Guided Decoding Issues

### "Guided decoding failed" or slow structured output

```bash
# Switch guided decoding backend
--guided-decoding-backend outlines    # default
--guided-decoding-backend lm-format-enforcer
```

### JSON schema too complex

Large/nested JSON schemas can slow generation. Simplify the schema or increase `--max-num-batched-tokens`.

## Speculative Decoding Issues

### No speedup

- Draft model too different from target — use a model from the same family
- Draft model too large — should be 5-10x smaller
- `--num-speculative-tokens` too high — try 3-5

### OOM with speculative decoding

Both models need to fit in memory:
```bash
# Use quantized draft model
vllm serve large-model --speculative-model small-model-AWQ --num-speculative-tokens 5
```

## Multimodal Issues

### Image processing errors

```bash
# Limit images per request
--limit-mm-per-prompt image=4

# Increase max model length for vision models (images consume many tokens)
--max-model-len 16384
```

## v1 Engine Issues

The v1 engine (`VLLM_USE_V1=1`) has different behavior:

- Some features not yet supported — check release notes
- Prefix caching behaves differently
- `--enforce-eager` may be needed for debugging

## Health Check and Monitoring

### Health endpoint

```bash
# Readiness check
curl http://localhost:8000/health
# Returns 200 when ready

# Detailed model info
curl http://localhost:8000/v1/models
```

### Key metrics to monitor

| Metric | Alert Threshold | Meaning |
|--------|----------------|---------|
| `vllm:num_requests_waiting` | > 50 | Requests queuing — scale up |
| `vllm:gpu_cache_usage_perc` | > 0.95 | KV cache nearly full |
| `vllm:time_to_first_token_seconds` p99 | > 5s | Prefill bottleneck |
| `vllm:time_per_output_token_seconds` p99 | > 100ms | Decode bottleneck |
| GPU utilization (DCGM) | < 50% | Batch too small or data starved |

### Grafana dashboard

Scrape `/metrics` with Prometheus, then import vLLM's community Grafana dashboards or build custom ones tracking the metrics above.
