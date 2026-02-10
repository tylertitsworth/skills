---
name: aiconfigurator
description: >
  NVIDIA AIConfigurator — optimal LLM serving configuration for disaggregated and aggregated deployments.
  Use when: (1) Finding optimal parallelism (TP/PP/EP/DP) for a model on specific GPUs,
  (2) Deciding between aggregated vs disaggregated serving for an SLA target,
  (3) Configuring prefill/decode worker ratios and batch sizes,
  (4) Generating Dynamo/K8s deployment configs from optimization results,
  (5) Tuning quantization settings (FP8, FP8-block, INT8, NVFP4) per component,
  (6) Planning MOE model deployments (expert parallelism, attention DP, wide EP),
  (7) Comparing aggregated vs disaggregated performance for vLLM on H100 clusters.
---

# AIConfigurator

AIConfigurator is an NVIDIA performance optimization tool that finds optimal LLM serving configurations. Given a model, GPU count, GPU type, and SLA targets, it searches thousands of configurations in seconds and generates ready-to-deploy config files.

- **GitHub**: [ai-dynamo/aiconfigurator](https://github.com/ai-dynamo/aiconfigurator)
- **Paper**: [arXiv:2601.06288](https://arxiv.org/abs/2601.06288)
- **Install**: `pip3 install aiconfigurator`

## Core Concepts

### What It Solves

Deploying LLMs requires choosing:
- **Aggregated vs Disaggregated** — should prefill and decode run on the same or separate workers?
- **Parallelism** — TP, PP, EP, DP settings per worker type
- **Worker ratios** — how many prefill vs decode workers (xPyD in disaggregated)
- **Batch sizes** — per-worker max batch size for SLA compliance
- **Quantization** — FP8, FP8-block, INT8, NVFP4 per component (GEMM, MOE, KV cache, attention)

AIConfigurator models LLM inference by breaking it into operations (GEMM, attention, communication, MOE), estimating execution time from collected hardware data, and searching the configuration space.

### How It Works

1. Break down inference into operations (GEMM, attention, all-reduce, all-to-all, etc.)
2. Look up operation latencies from collected silicon data for the target GPU
3. Compose end-to-end latency estimates for each configuration
4. Model continuous batching (aggregated) or disaggregated serving scheduling
5. Search all valid parallelism combinations against SLA constraints
6. Output ranked configurations with Pareto frontier visualization

### Key Terminology

| Term | Meaning |
|---|---|
| **Aggregated (agg)** | Prefill and decode share the same workers |
| **Disaggregated (disagg)** | Separate prefill and decode workers (xPyD) |
| **Replica** | Minimum scalable unit — in disagg: x prefill + y decode workers |
| **xPyD** | x prefill workers, y decode workers per replica |
| **ISL** | Input Sequence Length (prompt tokens) |
| **OSL** | Output Sequence Length (generated tokens) |
| **TTFT** | Time To First Token (ms) — prefill latency target |
| **TPOT** | Time Per Output Token (ms) — decode latency target |
| **MTP** | Multi-Token Prediction (speculative decoding with `nextn` draft tokens) |

## Installation

```bash
# PyPI
pip3 install aiconfigurator

# With webapp support
pip3 install aiconfigurator[webapp]

# From source (requires Git LFS for silicon data)
git clone https://github.com/ai-dynamo/aiconfigurator.git
git lfs pull
pip3 install .

# Docker
docker run -it --rm nvcr.io/nvidia/aiconfigurator:latest \
  aiconfigurator cli default --model LLAMA3.1_70B --total_gpus 16 --system h100_sxm --backend vllm
```

## CLI Reference

Four modes: `default`, `exp`, `generate`, `support`.

### `default` — Full Optimization Search

Compares aggregated vs disaggregated, searches all valid parallelism combinations, outputs ranked results with Pareto frontier.

```bash
aiconfigurator cli default \
  --model_path Qwen/Qwen3-32B \
  --total_gpus 32 \
  --system h100_sxm \
  --isl 4000 --osl 500 \
  --ttft 300 --tpot 10 \
  --backend vllm \
  --save_dir ./configs
```

| Flag | Required | Default | Purpose |
|---|---|---|---|
| `--model_path` / `--model` | ✅ | — | HuggingFace model ID or local path with `config.json` |
| `--total_gpus` | ✅ | — | Total GPUs available for deployment |
| `--system` | ✅ | — | GPU system: `h100_sxm`, `h200_sxm`, `a100_sxm`, `b200_sxm`, `gb200_sxm` |
| `--backend` | | `trtllm` | Inference backend: `vllm`, `trtllm`, `sglang`. **Use `vllm` for vLLM deployments.** |
| `--backend_version` | | latest | Specific backend version |
| `--isl` | | `4000` | Input sequence length (tokens) |
| `--osl` | | `1000` | Output sequence length (tokens) |
| `--prefix` | | `0` | Prefix cache length (shared prompt tokens) |
| `--ttft` | | `1000.0` | Target TTFT in ms (omit to leave unconstrained) |
| `--tpot` | | `40.0` | Target TPOT in ms (omit to leave unconstrained) |
| `--request_latency` | | — | End-to-end per-request latency limit (overrides `--tpot`) |
| `--save_dir` | | — | Directory for generated deployment configs |
| `--database_mode` | | `SILICON` | Performance data source (see below) |
| `--systems-paths` | | built-in | Override system YAML/data search paths (comma-separated) |

#### Database Modes

| Mode | Description |
|---|---|
| `SILICON` (default) | Uses collected silicon benchmark data — most accurate |
| `HYBRID` | Silicon when available, SOL+empirical for gaps |
| `EMPIRICAL` | SOL+empirical factor for all estimates |
| `SOL` | Speed-of-light theoretical only |

### `generate` — Quick Naive Config

Generates a working config without parameter sweep. Useful for quick deployment without SLA optimization.

```bash
aiconfigurator cli generate \
  --model_path Qwen/Qwen3-32B \
  --total_gpus 8 \
  --system h100_sxm \
  --backend vllm
```

Calculates minimum TP that fits: `TP * VRAM_per_GPU > 1.5 * model_weight_size`. No performance optimization.

### `exp` — Custom Experiments

Run customized experiments from a YAML config file. Control quantization, parallelism search space, and compare arbitrary configurations.

```bash
aiconfigurator cli exp --yaml_path my_experiments.yaml
```

See `references/experiment-config.md` for full YAML schema and `assets/` for example configs.

### `support` — Compatibility Check

Verify model + GPU + backend support:

```bash
aiconfigurator cli support \
  --model_path Qwen/Qwen3-32B \
  --system h100_sxm \
  --backend vllm
```

### Generator Flags (with `--save_dir`)

When `--save_dir` is specified, AIConfigurator generates deployment configs:

| Flag | Purpose |
|---|---|
| `--generator-config PATH` | YAML with ServiceConfig, K8sConfig, WorkerConfig sections |
| `--generator-set KEY=VALUE` | Inline overrides (repeatable) |
| `--generator-help` | Print full deployment schema and backend parameter mappings |
| `--generator-help deploy` | Print deployment YAML schema only |
| `--generator-help backend` | Print backend parameter mappings only |
| `--generated_config_version` | Generate configs for a different backend version than estimated |

```bash
# Generate configs with custom namespace and model path
aiconfigurator cli default \
  --model_path Qwen/Qwen3-32B --total_gpus 32 --system h100_sxm --backend vllm \
  --save_dir ./output \
  --generator-set K8sConfig.k8s_namespace=dynamo \
  --generator-set ServiceConfig.model_path=Qwen/Qwen3-32B-FP8
```

## Python API

```python
from aiconfigurator.cli import cli_default, cli_exp, cli_generate, cli_support

# Full optimization search
result = cli_default(
    model_path="Qwen/Qwen3-32B",
    total_gpus=32,
    system="h100_sxm",
    backend="vllm",
    ttft=300,
    tpot=10,
    isl=4000,
    osl=500,
)
print(result.best_configs["disagg"].head())

# Custom experiments from dict
result = cli_exp(config={
    "my_exp": {
        "serving_mode": "disagg",
        "model_path": "deepseek-ai/DeepSeek-V3",
        "total_gpus": 64,
        "system_name": "h100_sxm",
        "backend_name": "vllm",
        "isl": 4000,
        "osl": 1000,
    }
})

# Quick naive config
result = cli_generate(model_path="Qwen/Qwen3-32B", total_gpus=8, system="h100_sxm", backend="vllm")
print(result["parallelism"])  # {'tp': 1, 'pp': 1, 'replicas': 8, 'gpus_used': 8}

# Check support
agg, disagg = cli_support(model_path="Qwen/Qwen3-32B", system="h100_sxm", backend="vllm")
```

## Supported Hardware and Models

### GPU Systems

| System | GPUs | VRAM | vLLM Support |
|---|---|---|---|
| **`h100_sxm`** | **H100 SXM** | **80 GB** | **✅ Full support** |
| `h200_sxm` | H200 SXM | 141 GB | ✅ Full support |
| `a100_sxm` | A100 SXM | 80 GB | ✅ Supported |
| `b200_sxm` | B200 SXM | 192 GB | Preview |
| `gb200_sxm` | GB200 SXM | 384 GB | Preview |

### H100 Memory Budget (80 GB per GPU)

| Component | Typical Allocation |
|---|---|
| Model weights (bf16) | Depends on model / TP (e.g., 70B ÷ TP=4 ≈ 35 GB/GPU) |
| Model weights (FP8) | ~Half of bf16 |
| KV cache | Remaining after weights — controlled by `gpu_memory_utilization` |
| CUDA graphs + overhead | ~1-2 GB |

### Backend Framework Support

| Backend | Dense Models | MOE Models | Status |
|---|---|---|---|
| **`vllm`** | **✅** | **Dense only** | **Recommended — use `--backend vllm`** |
| `trtllm` (TensorRT-LLM) | ✅ | ✅ | Production (default) |
| `sglang` | ✅ | ✅ | Being evaluated |

### Supported Model Families

GPT, Llama 2/3, Qwen 2.5/3, Mixtral, DeepSeek-V3/R1. Also supports any HuggingFace model ID that falls into these families (non-MOE).

## Understanding Results

### Output Structure (with `--save_dir`)

```
results/
├── agg/
│   ├── best_config_topn.csv
│   ├── config.yaml
│   ├── pareto.csv
│   └── top1/
│       ├── agg/
│       │   ├── agg_config.yaml       # Framework config
│       │   ├── k8s_deploy.yaml        # K8s deployment manifest
│       │   └── node_0_run.sh          # Bare metal launch script
│       └── generator_config.yaml
├── disagg/
│   ├── best_config_topn.csv
│   ├── config.yaml
│   ├── pareto.csv
│   └── top1/
│       ├── disagg/
│       │   ├── prefill_config.yaml    # Prefill worker config
│       │   ├── decode_config.yaml     # Decode worker config
│       │   ├── k8s_deploy.yaml
│       │   └── node_0_run.sh
│       └── generator_config.yaml
└── pareto_frontier.png
```

### Reading the Results Table

The disagg results table columns:

| Column | Meaning |
|---|---|
| `tokens/s/gpu` | Aggregate throughput efficiency |
| `tokens/s/user` | Per-user generation speed (≈ 1000/TPOT) |
| `TTFT` | Estimated time to first token |
| `concurrency` | Total concurrent requests (= per-replica × replicas) |
| `replicas` | Number of xPyD replica units |
| `gpus/replica` | GPUs in each replica |
| `(p)workers` / `(d)workers` | Prefill / decode workers per replica |
| `(p)gpus/worker` / `(d)gpus/worker` | GPUs per prefill / decode worker |
| `(p)parallel` / `(d)parallel` | Parallelism config (e.g., `tp4pp1`) |
| `(p)bs` / `(d)bs` | Max batch size per worker |

### Pareto Frontier

The Pareto plot shows the trade-off between `tokens/s/gpu` (throughput) and `tokens/s/user` (per-user speed). Points on the frontier are optimal — you can't improve one without worsening the other. The `x` marker is the configuration that meets your SLA targets with the best throughput.

## Webapp

```bash
aiconfigurator webapp
# Visit http://127.0.0.1:7860
```

The webapp provides the same functionality as the CLI with a visual interface. All configurations and logic are identical.

## Cross-References

- [vllm](../vllm/) — vLLM engine configuration that AIConfigurator optimizes for
- [ray-serve](../ray-serve/) — Ray Serve PD deployment patterns aligned with AIConfigurator disagg configs

## Reference

- [AIConfigurator GitHub](https://github.com/ai-dynamo/aiconfigurator)
- [NVIDIA Dynamo docs](https://docs.nvidia.com/dynamo/latest/performance/aiconfigurator.html)
- [Paper: arXiv:2601.06288](https://arxiv.org/abs/2601.06288)
- `references/experiment-config.md` — full YAML experiment schema, quantization options, parallelism search space
- `references/advanced-tuning.md` — correction scales, replica config, practical search space reduction
- `assets/qwen3-32b-agg.yaml` / `assets/qwen3-32b-disagg.yaml` — Qwen3 32B (8–16× H100)
- `assets/qwen3-235b-agg.yaml` / `assets/qwen3-235b-disagg.yaml` — Qwen3-235B-A22B MoE FP8 (8–16× H100)
- `assets/glm-4.7-agg.yaml` / `assets/glm-4.7-disagg.yaml` — GLM-4.7 FP8 (8× H100)
- `assets/minimax-m1-agg.yaml` / `assets/minimax-m1-disagg.yaml` — MiniMax-M1 456B MoE experts_int8 (8–16× H100)
- `assets/kimi-k2-agg.yaml` / `assets/kimi-k2-disagg.yaml` — Kimi-K2 1T MoE FP8 (16–32× H100)
- `assets/llama-405b-agg.yaml` / `assets/llama-405b-disagg.yaml` — Llama 3.1 405B FP8 (16–32× H100)
- `assets/deepseek-v3-agg.yaml` / `assets/deepseek-v3-disagg.yaml` — DeepSeek-V3 671B MoE FP8 (32–64× H100)
