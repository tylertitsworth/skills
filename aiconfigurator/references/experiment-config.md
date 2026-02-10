# Experiment Configuration (YAML Schema)

The `exp` mode uses YAML files to define custom experiments. This allows fine-grained control over quantization, parallelism search space, and multi-experiment comparisons.

## Top-Level Structure

```yaml
# Optional: define execution order. If omitted, all experiments run.
exps:
  - experiment_name_1
  - experiment_name_2

# Each experiment is a top-level key
experiment_name_1:
  mode: "patch"              # Required: "patch" (merge with defaults) or "replace"
  serving_mode: "disagg"     # Required: "agg" or "disagg"
  model_path: "..."          # Required: HuggingFace model ID or local path
  total_gpus: 32             # Required: total GPUs for this experiment
  system_name: "h100_sxm"   # Required: GPU system name
  # ... optional fields and config section
```

## Common Fields

| Field | Required | Default | Purpose |
|---|---|---|---|
| `mode` | ✅ | — | `"patch"` merges with defaults; `"replace"` uses only what you define |
| `serving_mode` | ✅ | — | `"agg"` (aggregated) or `"disagg"` (disaggregated) |
| `model_path` | ✅ | — | HuggingFace model ID (e.g., `Qwen/Qwen3-32B`) or local path |
| `total_gpus` | ✅ | — | Total GPUs available |
| `system_name` | ✅ | — | GPU system: `h100_sxm`, `h100_sxm`, `a100_sxm`, `b200_sxm`, `gb200_sxm` |
| `decode_system_name` | | same as `system_name` | Different GPU type for decode (heterogeneous) |
| `backend_name` | | `trtllm` | Backend: `vllm`, `trtllm`, `sglang`. Use `vllm` for vLLM deployments. |
| `backend_version` | | latest | Specific version |
| `database_mode` | | `SILICON` | Data source: `SILICON`, `HYBRID`, `EMPIRICAL`, `SOL`, `SOL_FULL` |
| `isl` | | `4000` | Input sequence length |
| `osl` | | `1000` | Output sequence length |
| `prefix` | | `0` | Prefix cache length |
| `ttft` | | `1000.0` | TTFT target (ms) |
| `tpot` | | `40.0` | TPOT target (ms) |
| `enable_wideep` | | `false` | Wide expert parallelism for prefill/decode |
| `profiles` | | `[]` | Preset profiles to inherit |

## Config Section

The `config` key contains the search space and tuning parameters.

### Multi-Token Prediction (MTP / Speculative Decoding)

```yaml
config:
  nextn: 1                                    # Number of draft tokens (0 = disabled)
  nextn_accept_rates: [0.85, 0, 0, 0, 0]     # Accept rate per draft position
```

- `nextn: 1` means the model predicts 1 extra token speculatively
- `nextn_accept_rates[0] = 0.85` means 85% of first draft tokens are accepted
- DeepSeek models support `nextn: 2` with rates like `[0.85, 0.3, 0, 0, 0]`

### SGLang-Specific (Wide EP)

```yaml
config:
  moe_backend: null              # MOE backend for SGLang wide EP
  attention_backend: "flashinfer" # Attention backend for SGLang wide EP
```

## Worker Config (Quantization + Parallelism Search Space)

For aggregated mode, use `worker_config`. For disaggregated, use `prefill_worker_config` and `decode_worker_config`.

### Quantization Settings

```yaml
worker_config:
  gemm_quant_mode: "fp8_block"    # GEMM quantization
  moe_quant_mode: "fp8_block"     # MOE layer quantization
  kvcache_quant_mode: "float16"   # KV cache quantization
  fmha_quant_mode: "float16"      # Flash Multi-Head Attention quantization
  comm_quant_mode: "half"         # Communication quantization
```

| Setting | Options | Notes |
|---|---|---|
| `gemm_quant_mode` | `fp8`, `fp8_static`, `fp8_block`, `float16` | `fp8_block` is the current best for H100/H200 |
| `moe_quant_mode` | `fp8`, `fp8_block`, `w4afp8`, `float16` | `w4afp8` = INT4 weights + FP8 activations |
| `kvcache_quant_mode` | `fp8`, `int8`, `float16` | FP8 KV halves cache memory |
| `fmha_quant_mode` | `fp8`, `float16` | FP8 attention on H100+ |
| `comm_quant_mode` | `half` | Communication always in half precision |

#### Quantization Selection Guide

| Scenario | GEMM | MOE | KV Cache | FMHA |
|---|---|---|---|---|
| Maximum throughput | `fp8_block` | `fp8_block` | `fp8` | `fp8` |
| Quality-sensitive | `float16` | `float16` | `float16` | `float16` |
| Memory-constrained MOE | `fp8_block` | `w4afp8` | `fp8` | `float16` |
| Balanced | `fp8_block` | `fp8_block` | `float16` | `float16` |

### Parallelism Search Space

```yaml
worker_config:
  num_gpu_per_worker: [4, 8]      # Valid GPU counts per worker (exact match)
  tp_list: [1, 2, 4, 8]          # Tensor parallel options (attention module)
  pp_list: [1]                    # Pipeline parallel options (transformer layers)
  dp_list: [1, 2, 4, 8]          # Data parallel options (attention DP)
  moe_tp_list: [1]               # MOE tensor parallel options
  moe_ep_list: [1, 2, 4, 8]      # MOE expert parallel options
```

#### How Parallelism Enumeration Works

Valid configurations must satisfy:
```
tp × dp == moe_tp × moe_ep     # Attention and FFN use same GPU count
tp × dp × pp ∈ num_gpu_per_worker  # Total GPUs per worker matches list
```

For dense models (no MOE): `moe_tp_list` and `moe_ep_list` are ignored; `num_gpu_per_worker = tp × pp`.

#### Practical Search Space Reduction

For large models where you know the minimum TP:
```yaml
# DeepSeek-V3 (671B) — can't fit on < 8 GPUs per worker
worker_config:
  num_gpu_per_worker: [8]
  tp_list: [1]
  moe_ep_list: [8]
```

## Disaggregated-Specific Config

### Separate Prefill/Decode Worker Configs

```yaml
config:
  prefill_worker_config:
    gemm_quant_mode: "fp8_block"
    num_gpu_per_worker: [1, 2, 4]
    tp_list: [1, 2, 4]
    pp_list: [1]
    dp_list: [1]                 # Typically no attention DP for prefill
    moe_ep_list: [1, 2, 4]
  decode_worker_config:
    gemm_quant_mode: "fp8_block"
    num_gpu_per_worker: [2, 4, 8]
    tp_list: [1, 2, 4, 8]
    pp_list: [1]
    dp_list: [1, 2, 4, 8]       # Attention DP useful for decode throughput
    moe_ep_list: [1, 2, 4, 8]
```

### Replica Config

```yaml
config:
  replica_config:
    num_gpu_per_replica: [8, 16, 24, 32, 40, 48, 56, 64]
    max_gpu_per_replica: 128     # Cap (0 = no limit)
    max_prefill_worker: 32       # Max x in xPyD
    max_decode_worker: 32        # Max y in xPyD
```

| Field | Default | Purpose |
|---|---|---|
| `num_gpu_per_replica` | `[8, 16, ..., 128]` | Valid total GPU counts per replica. Aligns to server boundaries (multiples of 8). |
| `max_gpu_per_replica` | `128` | Hard cap. Caps the `num_gpu_per_replica` list. |
| `max_prefill_worker` | `32` | Maximum prefill workers per replica |
| `max_decode_worker` | `32` | Maximum decode workers per replica |

**Math**: `total_gpus_used = replicas × gpus_per_replica`. `gpus_per_replica = (p_workers × p_gpus_per_worker) + (d_workers × d_gpus_per_worker)`.

### Advanced Tuning Config

```yaml
config:
  advanced_tuning_config:
    prefill_latency_correction_scale: 1.1    # Scale predicted prefill latency
    decode_latency_correction_scale: 1.08    # Scale predicted decode latency
    prefill_max_batch_size: 1                # Max batch per prefill worker
    decode_max_batch_size: 512               # Max batch per decode worker
```

| Field | Default | Purpose |
|---|---|---|
| `prefill_latency_correction_scale` | `1.1` | Multiplier for predicted prefill latency. Increase if predictions are too optimistic. |
| `decode_latency_correction_scale` | `1.08` | Multiplier for predicted decode latency. |
| `prefill_max_batch_size` | `1` | For ISL > 1000, batch=1 already saturates compute. Larger batches multiply TTFT. |
| `decode_max_batch_size` | `512` | Per local rank. 512 is very high — most deployments need far less. |

## Multi-Experiment Comparison

Compare aggregated vs disaggregated, or different quantizations, in one YAML:

```yaml
exps:
  - baseline_agg
  - optimized_disagg
  - w4afp8_disagg

baseline_agg:
  mode: "patch"
  serving_mode: "agg"
  model_path: "deepseek-ai/DeepSeek-V3"
  total_gpus: 64
  system_name: "h100_sxm"

optimized_disagg:
  mode: "patch"
  serving_mode: "disagg"
  model_path: "deepseek-ai/DeepSeek-V3"
  total_gpus: 64
  system_name: "h100_sxm"

w4afp8_disagg:
  mode: "patch"
  serving_mode: "disagg"
  model_path: "deepseek-ai/DeepSeek-V3"
  total_gpus: 64
  system_name: "h100_sxm"
  config:
    prefill_worker_config:
      moe_quant_mode: "w4afp8"
    decode_worker_config:
      moe_quant_mode: "w4afp8"
```

## Heterogeneous Deployments

Use different GPU types for prefill and decode:

```yaml
hetero_experiment:
  mode: "patch"
  serving_mode: "disagg"
  model_path: "Qwen/Qwen3-32B"
  total_gpus: 16
  system_name: "h100_sxm"           # Prefill GPU type
  decode_system_name: "h100_sxm"    # Decode GPU type (different!)
```
