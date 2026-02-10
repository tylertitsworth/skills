# Mixture of Experts (MOE) Serving

Configuration and deployment of Mixture of Experts models in vLLM, covering expert parallelism, all2all communication backends, load balancing, and multi-node deployment.

## Expert Parallelism (EP)

Expert parallelism distributes different experts across GPUs instead of sharding each expert via tensor parallelism. Recommended for MOE models with enough GPUs.

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

The `all2all_backend` kwarg controls how tokens are routed between GPUs to reach their assigned experts.

| Backend | Description | Best For | Requirements |
|---|---|---|---|
| `"allgather_reducescatter"` | Default. AllGather + ReduceScatter | General purpose | None (works everywhere) |
| `"pplx"` | PPLX kernels | Single-node, high throughput | PPLX library |
| `"deepep_high_throughput"` | DeepEP high-throughput kernels | **Prefill** — large batch, compute-bound | DeepEP library |
| `"deepep_low_latency"` | DeepEP low-latency kernels | **Decode** — small batch, latency-sensitive | DeepEP library |
| `"flashinfer_all2allv"` | FlashInfer alltoallv | MNNVL (multi-node NVLink) | FlashInfer |
| `"naive"` | Broadcast-based | Debugging only | None |

```python
# Single node with pplx
llm = LLM(
    model="deepseek-ai/DeepSeek-V3-0324",
    enable_expert_parallel=True,
    tensor_parallel_size=1,
    data_parallel_size=8,
    all2all_backend="pplx",
)
```

### Backend Selection Guide

| Scenario | Recommended Backend | Why |
|---|---|---|
| Single node, 8 GPUs | `"pplx"` | Fastest single-node kernels |
| Multi-node, prefill-heavy | `"deepep_high_throughput"` | Optimized for large batches over network |
| Multi-node, decode/latency | `"deepep_low_latency"` | Minimizes per-token communication overhead |
| NVSwitch multi-node (DGX SuperPOD) | `"flashinfer_all2allv"` | Exploits NVLink fabric |
| Testing / no special libraries | `"allgather_reducescatter"` | Default, always works |

## Expert Parallelism Load Balancing (EPLB)

MOE models have uneven expert popularity — some experts handle more tokens. EPLB dynamically rebalances expert placement.

```python
llm = LLM(
    model="deepseek-ai/DeepSeek-V3-0324",
    enable_expert_parallel=True,
    data_parallel_size=8,
    enable_eplb=True,
    eplb_config={
        "window_size": 1000,
        "step_interval": 3000,
        "num_redundant_experts": 2,
        "log_balancedness": True,
    },
)
```

### EPLB Config Fields

| Field | Type | Default | Purpose |
|---|---|---|---|
| `window_size` | int | `1000` | Number of recent requests for load statistics |
| `step_interval` | int | `3000` | Rebalance every N steps |
| `num_redundant_experts` | int | `0` | Extra copies of hot experts. **Increase if some GPUs are overloaded.** Uses more memory but improves balance. |
| `log_balancedness` | bool | `False` | Log expert load distribution. **Enable for debugging imbalance.** |
| `use_async` | bool | `False` | Async rebalancing (reduces pause during rebalance) |
| `policy` | str | `"default"` | Rebalancing algorithm |

### Expert Placement Strategy

Controls initial expert distribution across GPUs:

```python
llm = LLM(
    model="...",
    enable_expert_parallel=True,
    expert_placement_strategy="round_robin",  # or "linear"
)
```

| Strategy | Distribution Example (4 experts, 2 GPUs) | Use Case |
|---|---|---|
| `"linear"` (default) | GPU 0: [0,1], GPU 1: [2,3] | Standard |
| `"round_robin"` | GPU 0: [0,2], GPU 1: [1,3] | **Better load balance for grouped expert models** (e.g., DeepSeek's shared expert groups) |

## Multi-Node MOE Deployment

For models that don't fit on a single node (DeepSeek-V3/R1, 671B parameters):

### Primary Node

```python
llm = LLM(
    model="deepseek-ai/DeepSeek-V3-0324",
    enable_expert_parallel=True,
    tensor_parallel_size=1,
    data_parallel_size=16,              # Total across all nodes
    data_parallel_size_local=8,         # GPUs on this node
    data_parallel_address="192.168.1.100",
    data_parallel_rpc_port=13345,
    all2all_backend="deepep_low_latency",
    api_server_count=8,                 # API server processes on primary
)
```

### Secondary Node (Headless)

```python
llm = LLM(
    model="deepseek-ai/DeepSeek-V3-0324",
    enable_expert_parallel=True,
    tensor_parallel_size=1,
    data_parallel_size=16,
    data_parallel_size_local=8,
    data_parallel_start_rank=8,         # Offset: ranks 8-15 on this node
    data_parallel_address="192.168.1.100",
    data_parallel_rpc_port=13345,
    all2all_backend="deepep_low_latency",
    headless=True,                      # Worker-only, no API server
)
```

### Multi-Node Configuration Reference

| Kwarg | Type | Purpose |
|---|---|---|
| `data_parallel_size` | int | Total DP workers across all nodes |
| `data_parallel_size_local` | int | DP workers on this node |
| `data_parallel_start_rank` | int | Starting rank for this node (0 for primary) |
| `data_parallel_address` | str | Primary node IP for coordination |
| `data_parallel_rpc_port` | int | RPC port for inter-node communication |
| `headless` | bool | Worker-only mode (no API server) |
| `api_server_count` | int | Number of API server processes on primary node |

## MOE + Disaggregated Serving

For production MOE deployments with strict SLA guarantees, combine EP with PD disaggregation. Prefill and decode benefit from different all2all backends:

```python
from vllm.config import KVTransferConfig

# MOE Prefill — high-throughput expert communication
prefill_llm = LLM(
    model="deepseek-ai/DeepSeek-V3-0324",
    enable_expert_parallel=True,
    tensor_parallel_size=1,
    data_parallel_size=8,
    all2all_backend="deepep_high_throughput",
    kv_transfer_config=KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_both",
        kv_buffer_device="cuda",
    ),
)

# MOE Decode — low-latency expert communication
decode_llm = LLM(
    model="deepseek-ai/DeepSeek-V3-0324",
    enable_expert_parallel=True,
    tensor_parallel_size=1,
    data_parallel_size=8,
    all2all_backend="deepep_low_latency",
    kv_transfer_config=KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_both",
        kv_buffer_device="cuda",
    ),
)
```

### Networking Considerations for MOE + PD

Two concurrent network traffic patterns:
1. **All2all traffic** — token routing between GPUs for expert computation
2. **KV transfer traffic** — KV cache between prefill and decode instances

On p5 instances (32 EFA, 3200 Gbps aggregate), bandwidth is sufficient for both. On p4d (4 EFA, 400 Gbps), contention is possible — monitor with `NCCL_DEBUG=INFO`. See [aws-efa](../../aws-efa/) for EFA configuration.

## Memory Planning

MOE models have unique memory characteristics due to large expert pools:

| Component | Estimate (DeepSeek-V3, bf16) |
|---|---|
| Shared layers (attention, norms) | ~20 GB |
| Expert parameters (256 experts) | ~600 GB total, distributed across EP GPUs |
| KV cache per request (8K ctx) | ~8 GB |
| All2all buffers | ~1-2 GB per GPU |

### Memory Budget Example: 8× H100 80GB, EP=8

| Component | Per GPU |
|---|---|
| Expert params (600 GB / 8) | ~75 GB |
| Shared layers | ~2.5 GB |
| All2all buffers | ~1.5 GB |
| **Remaining for KV cache** | **~1 GB** |

This is **extremely tight**. Solutions:

```python
# Option 1: FP8 quantization — halves expert memory
llm = LLM(
    model="deepseek-ai/DeepSeek-V3-0324",
    enable_expert_parallel=True,
    data_parallel_size=8,
    quantization="fp8",                    # ~37 GB experts/GPU instead of 75
    gpu_memory_utilization=0.95,
    kv_cache_dtype="fp8",                  # Also halve KV cache
)

# Option 2: More GPUs (16× H100, EP=16) — ~37 GB experts/GPU, more KV headroom
llm = LLM(
    model="deepseek-ai/DeepSeek-V3-0324",
    enable_expert_parallel=True,
    data_parallel_size=16,
    data_parallel_size_local=8,
)
```

## Troubleshooting

### All2All Communication Errors

**Symptom**: NCCL/pplx errors during expert routing.

| Check | Fix |
|---|---|
| GPU topology | `nvidia-smi topo -m` — NVLink preferred for intra-node all2all |
| Missing library | `pplx` requires PPLX; `deepep_*` requires DeepEP. Fall back to `allgather_reducescatter`. |
| Inter-node failures | Verify RDMA/EFA connectivity. Check security groups, placement groups. |

### Expert Load Imbalance

**Symptom**: Some GPUs at 100% utilization while others idle during MOE layers.

| Fix | Details |
|---|---|
| Enable EPLB logging | `eplb_config={"log_balancedness": True}` |
| Add redundant experts | `eplb_config={"num_redundant_experts": 2}` — copies hot experts |
| Round-robin placement | `expert_placement_strategy="round_robin"` |
| Increase EP degree | More GPUs = finer expert distribution |

### OOM with MOE Models

| Fix | Details |
|---|---|
| FP8 quantization | `quantization="fp8"` — halves expert memory |
| FP8 KV cache | `kv_cache_dtype="fp8"` — halves KV cache memory |
| Increase GPU count | More GPUs for EP = less expert memory per GPU |
| Reduce context | Lower `max_model_len` if KV cache OOM during serving |
