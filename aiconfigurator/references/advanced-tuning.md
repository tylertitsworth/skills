# Advanced Tuning

Beyond the default CLI, AIConfigurator exposes detailed tuning controls for quantization per component, parallelism search space, correction factors, and deployment generation.

## When to Use Advanced Tuning

| Scenario | Approach |
|---|---|
| Quick deployment, no special requirements | `aiconfigurator cli default` — defaults work well |
| Specific quantization requirements (e.g., FP16 for quality) | `exp` mode with custom `worker_config` |
| Known hardware constraints (e.g., can't use TP > 4) | Restrict `tp_list` in worker config |
| Predictions too optimistic | Increase `latency_correction_scale` |
| Wide EP for large MOE models | Set `enable_wideep: true` or increase `max_gpu_per_replica` |
| Compare multiple strategies | Multi-experiment YAML with `exps` list |

## Practical Search Space Reduction

Reducing the search space saves time and avoids invalid configurations:

### Dense Models (Llama, Qwen)

```yaml
worker_config:
  num_gpu_per_worker: [1, 2, 4, 8]
  tp_list: [1, 2, 4, 8]
  pp_list: [1]           # PP rarely helps for serving (adds pipeline bubbles)
  dp_list: [1]           # No attention DP for dense models
  moe_tp_list: [1]       # Irrelevant for dense
  moe_ep_list: [1]       # Irrelevant for dense
```

### Large Dense Models (405B+)

```yaml
# Llama 3.1 405B on H100 80 GB — 405B bf16 ≈ 810 GB, needs TP=8 minimum
# At FP8: ~405 GB → TP=8 gives ~51 GB/GPU, leaving ~27 GB for KV cache
worker_config:
  num_gpu_per_worker: [8]
  tp_list: [8]
  pp_list: [1, 2]        # PP useful for 405B if KV cache is too tight at TP=8
```

### MOE Models (DeepSeek-V3, Mixtral)

```yaml
# DeepSeek-V3 on H100 80 GB — 671B total, ~335 GB at FP8
# EP=8: ~42 GB experts/GPU + ~10 GB shared = ~52 GB, leaving ~26 GB for KV
prefill_worker_config:
  num_gpu_per_worker: [8]
  tp_list: [1]
  moe_ep_list: [8]       # EP=8 distributes 256 experts across GPUs
  dp_list: [1]
decode_worker_config:
  num_gpu_per_worker: [4, 8]
  tp_list: [1, 2, 4]
  moe_ep_list: [2, 4, 8]
  dp_list: [1, 2, 4]     # Attention DP for decode throughput
```

## Correction Factors

If actual deployment performance doesn't match AIConfigurator estimates:

```yaml
advanced_tuning_config:
  prefill_latency_correction_scale: 1.15   # Increase if TTFT is worse than predicted
  decode_latency_correction_scale: 1.10    # Increase if TPOT is worse than predicted
```

**How to calibrate**: Run a quick benchmark on your target hardware. If measured TTFT is 15% higher than predicted, set `prefill_latency_correction_scale: 1.15`.

## Batch Size Tuning

```yaml
advanced_tuning_config:
  prefill_max_batch_size: 1       # Almost always 1 for ISL > 1000
  decode_max_batch_size: 256      # Reduce from default 512 for stricter TPOT
```

- **Prefill batch size**: For ISL > 1000 tokens, a single prefill already saturates GPU compute. Batching prefills multiplies TTFT linearly with minimal throughput gain. Keep at 1.
- **Decode batch size**: Higher = more throughput but higher TPOT. Reduce for stricter latency targets.

## Generator Configuration

When using `--save_dir`, control the generated deployment configs:

### ServiceConfig

```bash
--generator-set ServiceConfig.model_path=Qwen/Qwen3-32B-FP8
--generator-set ServiceConfig.model_name=qwen3-32b
```

### K8sConfig

```bash
--generator-set K8sConfig.k8s_namespace=dynamo
--generator-set K8sConfig.k8s_image=nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3
```

### Rule Plugins

```bash
# Production rules (default) — adjusted batch sizes and CUDA graph sizes
aiconfigurator cli default ... --save_dir ./output

# Benchmark rules — match simulated results more closely
aiconfigurator cli default ... --save_dir ./output --generator-set rule=benchmark
```

## Performance Estimation Model

Understanding how AIConfigurator estimates performance helps interpret results:

### Operation Breakdown

| Operation | Prefill Impact | Decode Impact |
|---|---|---|
| GEMM | ★★★ (compute-bound) | ★★ (memory-bound) |
| Attention (FMHA) | ★★★ | ★★ |
| AllReduce (TP) | ★★ | ★ |
| All-to-All (EP) | ★★ (many tokens) | ★★ (per-token) |
| P2P (PP) | ★ | ★ |
| KV Cache I/O | ★ | ★★★ (memory-bound bottleneck) |

### Limitations

- Memory estimation is approximate — always validate with actual deployment
- Results can be overly optimistic in the low-latency, high-throughput region
- vLLM and SGLang backends are being evaluated — validate with real benchmarks
- Cross-node communication overhead may differ from silicon data (collected on single-node)
