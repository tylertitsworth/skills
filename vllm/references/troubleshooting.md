# vLLM Troubleshooting

All fixes are configuration changes — update the container args or `LLM()` kwargs and redeploy.

## GPU OOM (Out of Memory)

### During model loading

The model weights don't fit in GPU memory.

**Fixes (ordered by preference):**

1. **Enable quantization** — add `quantization="fp8"` (8-bit, minimal quality loss) or `quantization="awq"` (4-bit, needs pre-quantized model)
2. **Use tensor parallelism** — set `tensor_parallel_size=2` (or more) to shard across GPUs
3. **Reduce max model length** — set `max_model_len=4096` to reduce KV cache allocation
4. **Lower GPU memory utilization** — set `gpu_memory_utilization=0.85` (counterintuitively helps with fragmentation)

### During inference

KV cache is full — too many concurrent sequences or long contexts.

**Fixes:**

1. Reduce `max_num_seqs=64` (fewer concurrent requests)
2. Reduce `max_model_len=4096`
3. Enable `enable_prefix_caching=True` (reuse KV cache for shared prefixes)
4. Enable `enable_chunked_prefill=True` (process long prompts in chunks)

### Memory estimation

```
# Rough model memory (weights only):
# Parameters × bytes_per_param
# 7B × 2 bytes (fp16) = ~14 GB
# 70B × 2 bytes (fp16) = ~140 GB
# 70B × 1 byte (fp8) = ~70 GB

# KV cache per token per layer (fp16):
# 2 × num_heads × head_dim × 2 bytes × num_layers
# Llama-3.1-8B: 2 × 32 × 128 × 2 × 32 = ~512 KB per token
```

## Slow Time-to-First-Token (TTFT)

TTFT is the time from request receipt to first generated token. Slow TTFT usually means the prefill phase is bottlenecked.

**Diagnosis:** Check `vllm:time_to_first_token_seconds` in Prometheus metrics (scraped from the `/metrics` endpoint via ServiceMonitor).

**Fixes (config changes):**

1. `enable_chunked_prefill=True` — allows decode steps during long prefills
2. `max_num_batched_tokens=4096` — limits prefill batch size
3. `enable_prefix_caching=True` — skips re-processing shared prefixes
4. `quantization="fp8"` — faster compute
5. `speculative_model="small-model"`, `num_speculative_tokens=5` — improves decode speed

## Throughput Degradation

Throughput drops as concurrency increases.

**Diagnosis:** Check Prometheus metrics:
- `vllm:num_requests_waiting` — requests queuing
- `vllm:gpu_cache_usage_perc` — KV cache utilization

**Fixes:**

| Symptom | Metric | Fix |
|---|---|---|
| Requests queuing | `num_requests_waiting` high | Increase `gpu_memory_utilization` or reduce `max_model_len` |
| GPU underutilized | Low GPU util | Increase `max_num_seqs=512` |
| Long prompts dominate | High TTFT variance | Enable `enable_chunked_prefill=True` |
| Single replica saturated | All metrics high | Add replicas via HPA on `vllm:num_requests_waiting` |

## Model Loading Failures

### "ValueError: The model's max seq len is too large"

Set `max_model_len=8192` (or smaller) to override the model's default context length.

### Authentication errors / model not found

- Set `HF_TOKEN` env var in the container spec for gated models (Llama, Gemma, etc.)
- For air-gapped clusters, use an init container or PVC to pre-download models and set `model="/path/to/local/model"`

### "torch.cuda.OutOfMemoryError" during loading

Model too large for available GPUs. Set `tensor_parallel_size=2` or `quantization="fp8"`.

### Custom code models

Set `trust_remote_code=True` for models with custom architectures.

## Multi-GPU Communication Errors

### Worker crashed ("remote process exited")

1. Check GPU resource requests match actual node capacity
2. Ensure `/dev/shm` volume is mounted (required for multi-GPU):
   ```yaml
   volumes:
   - name: shm
     emptyDir:
       medium: Memory
       sizeLimit: 8Gi
   ```
3. Verify GPU driver version consistency across nodes (for multi-node)
4. Check that `resources.limits.nvidia.com/gpu` matches `tensor_parallel_size`

## CUDA Graph Issues

### "CUDA error: out of memory" during warmup

CUDA graphs pre-allocate memory during warmup. Set `enforce_eager=True` to disable (slower but less memory), or reduce `max_num_seqs` to limit graph variants.

### Crashes with dynamic shapes

CUDA graphs require static shapes. Set `enforce_eager=True` to fall back to eager mode.

## LoRA Adapter Issues

### "LoRA adapter not found"

Verify the adapter path is accessible from the container (mounted via PVC or ConfigMap). Ensure `enable_lora=True` is set.

### Too many adapters / OOM

Limit concurrent adapters: `max_loras=4`, `max_lora_rank=64`, `lora_extra_vocab_size=256`.

## Structured Output / Guided Decoding Issues

### Slow or failed guided decoding

Switch backend: `guided_decoding_backend="lm-format-enforcer"` (default is `outlines`). Large/nested JSON schemas can slow generation — simplify the schema.

## Speculative Decoding Issues

### No speedup

- Draft model too different from target — use same model family
- Draft model too large — should be 5-10x smaller
- `num_speculative_tokens` too high — try 3-5

### OOM with speculative decoding

Both models must fit in GPU memory. Use a quantized draft model or increase `tensor_parallel_size`.

## Multimodal Issues

Set `limit_mm_per_prompt={"image": 4}` to limit images per request. Vision models consume many tokens per image — increase `max_model_len` accordingly.

## v1 Engine Issues

The v1 engine (`VLLM_USE_V1=1` env var) has different behavior:

- Some features not yet supported — check release notes
- Prefix caching behaves differently
- Set `enforce_eager=True` for debugging

## Key Metrics to Monitor

| Metric | Alert Threshold | Meaning |
|--------|----------------|---------|
| `vllm:num_requests_waiting` | > 50 | Requests queuing — scale up |
| `vllm:gpu_cache_usage_perc` | > 0.95 | KV cache nearly full |
| `vllm:time_to_first_token_seconds` p99 | > 5s | Prefill bottleneck |
| `vllm:time_per_output_token_seconds` p99 | > 100ms | Decode bottleneck |
| GPU utilization (DCGM) | < 50% | Batch too small or data starved |

Scrape `/metrics` via Prometheus ServiceMonitor. Build Grafana dashboards on the metrics above.
