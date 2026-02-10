# Mixture of Experts (MOE) Serving

Configuration and deployment of Mixture of Experts models in vLLM, covering expert parallelism, all2all communication backends, load balancing, and multi-node deployment.

## Expert Parallelism (EP)

Expert parallelism distributes different experts across GPUs instead of sharding each expert via tensor parallelism. This is the recommended parallelism for MOE models when you have enough GPUs.

```python
from vllm import LLM

llm = LLM(
    model="deepseek-ai/DeepSeek-V3-0324",
    enable_expert_parallel=True,       # EP instead of TP for MOE layers
    tensor_parallel_size=1,            # TP=1 per expert (EP handles distribution)
    data_parallel_size=8,              # 8-way DP across GPUs
)
```

### When to Use EP vs TP

| Approach | Use When | Trade-off |
|---|---|---|
| **TP only** | Few GPUs, dense model layers dominate | Each GPU holds a shard of every expert |
| **EP only** (`enable_expert_parallel=True`, `TP=1`) | Many GPUs, MOE layers dominate | Each GPU owns full experts, needs all2all for routing |
| **EP + TP** | Very large experts that don't fit on one GPU | Combined sharding; highest communication overhead |
| **EP + DP** | Production serving at scale | Data parallelism across EP groups for throughput |

### Key Principle

With `enable_expert_parallel=True`, vLLM uses the tensor parallel degree to determine EP degree. Setting `tensor_parallel_size=1` and `data_parallel_size=8` means 8-way expert parallelism where each GPU owns a subset of experts.

## All2All Communication Backends

The all2all backend controls how tokens are routed between GPUs to reach their assigned experts. Choice significantly impacts performance.

| Backend | Description | Best For |
|---|---|---|
| `allgather_reducescatter` | Default. AllGather + ReduceScatter | General purpose, works everywhere |
| `pplx` | PPLX kernels | Single-node, high throughput |
| `deepep_high_throughput` | DeepEP high-throughput kernels | **Prefill** — large batch, compute-bound |
| `deepep_low_latency` | DeepEP low-latency kernels | **Decode** — small batch, latency-sensitive |
| `flashinfer_all2allv` | FlashInfer alltoallv | MNNVL (multi-node NVLink) |
| `naive` | Broadcast-based | Debugging only |

```bash
# Single node, high throughput
vllm serve deepseek-ai/DeepSeek-V3-0324 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend pplx

# Multi-node with DeepEP
vllm serve deepseek-ai/DeepSeek-V3-0324 \
  --tensor-parallel-size 1 \
  --data-parallel-size 16 \
  --data-parallel-size-local 8 \
  --enable-expert-parallel \
  --all2all-backend deepep_high_throughput
```

## Expert Parallelism Load Balancing (EPLB)

MOE models have uneven expert popularity — some experts handle more tokens than others. EPLB dynamically rebalances expert placement across GPUs.

```bash
vllm serve deepseek-ai/DeepSeek-V3-0324 \
  --enable-expert-parallel \
  --enable-eplb \
  --eplb-config '{
    "window_size": 1000,
    "step_interval": 3000,
    "num_redundant_experts": 2,
    "log_balancedness": true
  }'
```

### EPLB Config Fields

| Field | Default | Purpose |
|---|---|---|
| `window_size` | `1000` | Number of recent requests to consider for load stats |
| `step_interval` | `3000` | Rebalance every N steps |
| `num_redundant_experts` | `0` | Extra expert copies for hot experts. Increase if some GPUs are overloaded. |
| `log_balancedness` | `false` | Log expert load distribution for debugging |
| `use_async` | `false` | Async rebalancing (reduces pause) |
| `policy` | `default` | Rebalancing policy |

### Expert Placement Strategy

Controls how experts are initially distributed across GPUs:

```bash
--expert-placement-strategy round_robin  # or "linear"
```

| Strategy | Distribution | Use Case |
|---|---|---|
| `linear` | GPU 0 gets experts [0,1], GPU 1 gets [2,3] | Default |
| `round_robin` | GPU 0 gets experts [0,2], GPU 1 gets [1,3] | Better load balance for grouped expert models |

## Multi-Node Deployment

For large MOE models (DeepSeek-V3/R1, 671B parameters) that don't fit on a single node:

### Primary Node

```bash
vllm serve deepseek-ai/DeepSeek-V3-0324 \
  --all2all-backend deepep_low_latency \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --data-parallel-size 16 \
  --data-parallel-size-local 8 \
  --data-parallel-address 192.168.1.100 \
  --data-parallel-rpc-port 13345 \
  --api-server-count 8
```

### Secondary Node (Headless)

```bash
vllm serve deepseek-ai/DeepSeek-V3-0324 \
  --all2all-backend deepep_low_latency \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --data-parallel-size 16 \
  --data-parallel-size-local 8 \
  --data-parallel-start-rank 8 \
  --data-parallel-address 192.168.1.100 \
  --data-parallel-rpc-port 13345 \
  --headless
```

### Multi-Node Config Reference

| Flag | Purpose |
|---|---|
| `--data-parallel-size` | Total DP workers across all nodes |
| `--data-parallel-size-local` | DP workers on this node |
| `--data-parallel-start-rank` | Starting rank for this node (0 for primary) |
| `--data-parallel-address` | Primary node IP for coordination |
| `--data-parallel-rpc-port` | RPC port for inter-node communication |
| `--headless` | Worker-only mode (no API server) |
| `--api-server-count` | Number of API server processes on primary |

## MOE + Disaggregated Serving

For production MOE deployments requiring strict SLA guarantees, combine expert parallelism with prefill-decode disaggregation:

- **Prefill instances** use `deepep_high_throughput` — optimized for large batches where many experts activate
- **Decode instances** use `deepep_low_latency` — optimized for small per-token batches with sparse activation

```bash
# MOE Prefill instance
vllm serve deepseek-ai/DeepSeek-V3-0324 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend deepep_high_throughput \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'

# MOE Decode instance
vllm serve deepseek-ai/DeepSeek-V3-0324 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend deepep_low_latency \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
```

### Networking Considerations

MOE + PD creates two overlapping network traffic patterns:
1. **All2all traffic** — token routing between GPUs for expert computation (intra-node or inter-node)
2. **KV transfer traffic** — KV cache transfer between prefill and decode instances

For multi-node setups, ensure sufficient bandwidth for both. On AWS, this means EFA with multiple ENIs. See [aws-efa](../../aws-efa/) for EFA configuration with disaggregated serving.

## Memory Planning

MOE models have unique memory characteristics:

| Component | Size Estimate (DeepSeek-V3, bf16) |
|---|---|
| Shared layers (attention, norms) | ~20 GB |
| Expert parameters (256 experts) | ~600 GB total, distributed across EP GPUs |
| KV cache per request (8K ctx) | ~5 GB |
| All2all buffers | ~1-2 GB per GPU |

With EP=8 on 8× H100 (80 GB each):
- Expert params per GPU: ~75 GB
- Available for KV cache: ~3-4 GB per GPU
- **This is tight** — use FP8 quantization or increase GPU count

```python
llm = LLM(
    model="deepseek-ai/DeepSeek-V3-0324",
    enable_expert_parallel=True,
    data_parallel_size=8,
    quantization="fp8",                # halves expert memory
    gpu_memory_utilization=0.95,
)
```

## Troubleshooting

### All2All Communication Errors

**Symptom**: NCCL/pplx errors during expert routing.

1. Check GPU topology: `nvidia-smi topo -m`. NVLink preferred for intra-node all2all
2. For `pplx` backend: requires PPLX library installed. Fall back to `allgather_reducescatter` if unavailable
3. For `deepep_*` backends: requires DeepEP library. Check `pip list | grep deepep`

### Expert Load Imbalance

**Symptom**: Some GPUs at 100% utilization while others idle during MOE layers.

1. Enable EPLB logging: `--eplb-config '{"log_balancedness":true}'`
2. Increase redundant experts: `"num_redundant_experts": 2` or higher
3. Try `round_robin` placement: `--expert-placement-strategy round_robin`
4. For persistent imbalance, increase `data_parallel_size` (more GPUs = finer expert distribution)

### OOM with MOE Models

Expert parameters are large. If OOM during loading:
1. Use FP8 quantization: `quantization="fp8"`
2. Increase EP degree (more GPUs share the expert pool)
3. Reduce `gpu_memory_utilization` temporarily to diagnose
4. For KV cache OOM during serving: reduce `max_model_len` or increase decode replicas
