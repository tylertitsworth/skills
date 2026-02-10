# Disaggregated Prefill-Decode Serving

Disaggregated prefill-decode (PD) serving separates the prefill phase (processing input prompts) and the decode phase (generating tokens) onto different vLLM instances for independent scaling and latency optimization.

## Why Disaggregated Serving?

1. **Independent TTFT and ITL tuning** — assign different parallelism strategies to prefill and decode instances
2. **Controlled tail latency** — decode instances are never interrupted by prefill work, eliminating ITL spikes
3. **Hardware optimization** — prefill is compute-bound (benefits from larger TP), decode is memory-bound (benefits from more KV cache)
4. **Independent scaling** — scale prefill instances for prompt-heavy workloads, decode instances for generation-heavy ones

> **Note:** Disaggregated prefill does NOT improve aggregate throughput. It improves latency characteristics and SLA compliance.

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              Request Router              │
                    │    (vLLM proxy / Ray Serve / custom)     │
                    └──────┬──────────────────────┬───────────┘
                           │                      │
                    ┌──────▼──────┐        ┌──────▼──────┐
                    │   Prefill   │        │   Prefill   │
                    │  Instance 0 │        │  Instance 1 │
                    │  (TP=4, A100)│       │  (TP=4, A100)│
                    └──────┬──────┘        └──────┬──────┘
                           │ KV Transfer           │ KV Transfer
                           │ (NixlConnector)       │ (NixlConnector)
                    ┌──────▼──────┐        ┌──────▼──────┐
                    │   Decode    │        │   Decode    │
                    │  Instance 0 │        │  Instance 1 │
                    │  (TP=2, A100)│       │  (TP=2, A100)│
                    └─────────────┘        └─────────────┘
```

Two vLLM instance types run simultaneously:
- **Prefill instance** (`kv_role="kv_producer"` or `"kv_both"`) — processes prompts, produces KV caches, compute-bound
- **Decode instance** (`kv_role="kv_consumer"` or `"kv_both"`) — receives KV caches, generates tokens, memory-bound
- **Router/proxy** — directs requests: prefill phase to prefill instances, decode phase to decode instances

## Connector Types

| Connector | Transport | Use Case |
|---|---|---|
| `NixlConnector` | NIXL (RDMA/UCX/GDS) | **Production** — fully async, highest performance |
| `LMCacheConnectorV1` | LMCache + NIXL | Production — with KV cache sharing/offloading |
| `P2pNcclConnector` | NCCL P2P | Same-node GPU-to-GPU transfer |
| `MooncakeConnector` | Mooncake | Alternative transport |
| `OffloadingConnector` | CPU memory | KV offloading to CPU |
| `MultiConnector` | Multiple | Chain multiple connectors |

### Choosing a Connector

- **Cross-node, production**: `NixlConnector` with UCX backend (RDMA if available)
- **Cross-node with cache sharing**: `LMCacheConnectorV1` (adds dedup, CPU offload)
- **Same node**: `P2pNcclConnector` (GPU direct) or `NixlConnector` with shared memory
- **Hybrid**: `MultiConnector` to chain NixlConnector + OffloadingConnector

## NixlConnector Configuration

### Basic Setup

```python
from vllm.config import KVTransferConfig

# Prefill instance
KVTransferConfig(
    kv_connector="NixlConnector",
    kv_role="kv_producer",         # or "kv_both" for bidirectional
    kv_buffer_device="cuda",       # "cuda" (GPU RDMA) or "cpu"
)

# Decode instance
KVTransferConfig(
    kv_connector="NixlConnector",
    kv_role="kv_consumer",         # or "kv_both"
    kv_buffer_device="cuda",
)
```

### Extra Config Options

```python
KVTransferConfig(
    kv_connector="NixlConnector",
    kv_role="kv_both",
    kv_buffer_device="cuda",
    kv_connector_extra_config={
        "backends": ["UCX"],           # NIXL backends: "UCX", "GDS"
    },
)
```

### Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `VLLM_NIXL_SIDE_CHANNEL_PORT` | `5600` | Port for NIXL handshake. Each worker needs unique port on same host. With DP, port = `base_port + dp_rank`. |
| `VLLM_NIXL_SIDE_CHANNEL_HOST` | `localhost` | Host for side channel. **Set when prefill/decode are on different machines.** |
| `VLLM_NIXL_ABORT_REQUEST_TIMEOUT` | `480` | Seconds before releasing KV cache blocks for aborted requests |
| `UCX_TLS` | — | UCX transports: `all`, or specific like `rc,ud,sm` |
| `UCX_NET_DEVICES` | — | Network devices: `all`, or specific like `mlx5_0:1,mlx5_1:1` |

### CLI Usage

```bash
# Prefill instance
CUDA_VISIBLE_DEVICES=0,1,2,3 \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --port 8100 \
  --tensor-parallel-size 4 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'

# Decode instance (different node or GPU set)
CUDA_VISIBLE_DEVICES=4,5 \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --port 8200 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
```

## KVTransferConfig Reference

| Field | Type | Default | Purpose |
|---|---|---|---|
| `kv_connector` | str | `None` | Connector class name |
| `kv_role` | str | `None` | `kv_producer`, `kv_consumer`, `kv_both` |
| `kv_buffer_device` | str | `cuda` | Buffer device: `cuda` or `cpu` |
| `kv_buffer_size` | float | `1e9` | Buffer size in bytes (TorchDistributedConnector) |
| `kv_rank` | int | `None` | Instance rank (0=prefill, 1=decode for P2pNccl) |
| `kv_parallel_size` | int | `1` | Parallel instances (2 for P2pNcclConnector) |
| `kv_ip` | str | `127.0.0.1` | IP for distributed connection |
| `kv_port` | int | `14579` | Port for distributed connection |
| `kv_connector_extra_config` | dict | `{}` | Connector-specific config |
| `kv_load_failure_policy` | str | `recompute` | `recompute` (retry locally) or `fail` (error) |

## Request Routing

### vLLM Disaggregated Proxy

vLLM provides an example proxy server that routes OpenAI-compatible requests between prefill and decode instances:

```python
from openai import OpenAI
import uuid

prefill_client = OpenAI(base_url="http://prefill-host:8100/v1", api_key="EMPTY")
decode_client = OpenAI(base_url="http://decode-host:8200/v1", api_key="EMPTY")

request_id = str(uuid.uuid4())
model = prefill_client.models.list().data[0].id

# Phase 1: Prefill — produces KV cache, returns transfer params
prefill_resp = prefill_client.completions.create(
    model=model,
    prompt="Explain the architecture of transformer models in detail",
    max_tokens=1,
    extra_body={
        "kv_transfer_params": {
            "do_remote_decode": True,
            "do_remote_prefill": False,
        }
    },
    extra_headers={"X-Request-Id": request_id},
)

# Phase 2: Decode — uses transferred KV cache
decode_resp = decode_client.completions.create(
    model=model,
    prompt="ignored",
    max_tokens=512,
    extra_body={
        "kv_transfer_params": prefill_resp.kv_transfer_params,
    },
    extra_headers={"X-Request-Id": request_id},
)
```

### Ray Serve PD Integration

Ray Serve provides `build_pd_openai_app` for native PD routing — see [ray-serve](../../ray-serve/) skill.

## OffloadingConnector

Offload KV cache to CPU memory (reduces GPU memory pressure):

```python
KVTransferConfig(
    kv_connector="OffloadingConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "block_size": 64,
        "cpu_bytes_to_use": 10_000_000_000,  # 10 GB CPU memory
    },
)
```

## MultiConnector (Chaining)

Combine connectors — e.g., NIXL for cross-node transfer + CPU offloading:

```python
KVTransferConfig(
    kv_connector="MultiConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "connectors": [
            {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
            {"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
             "kv_connector_extra_config": {"block_size": 64, "cpu_bytes_to_use": 10_000_000_000}},
        ],
    },
)
```

## Kubernetes Deployment

Deploy prefill and decode as separate Deployments with shared model and different resource profiles:

```yaml
# Prefill Deployment — compute-optimized, higher TP
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-prefill
  labels:
    app: vllm
    role: prefill
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm
      role: prefill
  template:
    metadata:
      labels:
        app: vllm
        role: prefill
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        env:
        - name: VLLM_NIXL_SIDE_CHANNEL_PORT
          value: "5600"
        - name: UCX_NET_DEVICES
          value: "all"
        args:
        - --model=meta-llama/Llama-3.1-70B-Instruct
        - --tensor-parallel-size=4
        - --port=8100
        - --kv-transfer-config={"kv_connector":"NixlConnector","kv_role":"kv_producer"}
        resources:
          limits:
            nvidia.com/gpu: "4"
        ports:
        - containerPort: 8100
        - containerPort: 5600    # NIXL side channel
---
# Decode Deployment — memory-optimized, more KV cache headroom
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-decode
  labels:
    app: vllm
    role: decode
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm
      role: decode
  template:
    metadata:
      labels:
        app: vllm
        role: decode
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        env:
        - name: VLLM_NIXL_SIDE_CHANNEL_PORT
          value: "5601"
        - name: UCX_NET_DEVICES
          value: "all"
        args:
        - --model=meta-llama/Llama-3.1-70B-Instruct
        - --tensor-parallel-size=2
        - --gpu-memory-utilization=0.95
        - --port=8200
        - --kv-transfer-config={"kv_connector":"NixlConnector","kv_role":"kv_consumer"}
        resources:
          limits:
            nvidia.com/gpu: "2"
        ports:
        - containerPort: 8200
        - containerPort: 5601
```

## Disaggregated Serving with MOE Models

MOE models benefit strongly from PD disaggregation because:
- **Prefill phase** activates many experts per token (compute-bound) — benefits from `deepep_high_throughput` all2all backend
- **Decode phase** is memory-bound with sparse expert activation — benefits from `deepep_low_latency` backend

```bash
# Prefill instance — high-throughput expert communication
vllm serve deepseek-ai/DeepSeek-V3-0324 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend deepep_high_throughput \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'

# Decode instance — low-latency expert communication
vllm serve deepseek-ai/DeepSeek-V3-0324 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend deepep_low_latency \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
```

See `references/moe-serving.md` for complete MOE/EP configuration.

## Troubleshooting

### KV Cache Transfer Failures

**Symptom**: Decode instance hangs waiting for KV cache, or requests time out.

Checks:
1. Verify NIXL side channel connectivity: both instances must reach each other on `VLLM_NIXL_SIDE_CHANNEL_PORT`
2. Check `UCX_NET_DEVICES` — must match available network interfaces (`ibstat` for InfiniBand, `ip link` for Ethernet)
3. Both instances **must use the same model, tokenizer, and block size**
4. If using `kv_role="kv_producer"` / `"kv_consumer"`, ensure roles are assigned correctly (not swapped)
5. Check firewall rules allow both the NIXL side channel port AND the data transfer ports (UCX uses ephemeral ports)

**Fix**: Start with `kv_role="kv_both"` and `--enforce-eager` to rule out CUDA graph issues, then narrow down.

### High Transfer Latency

**Symptom**: TTFT is worse than monolithic serving despite PD split.

Causes:
1. **Network bottleneck** — KV cache transfer over TCP instead of RDMA. Check `UCX_TLS` includes `rc` (reliable connected) for InfiniBand
2. **CPU buffer path** — `kv_buffer_device="cpu"` adds a GPU→CPU→GPU copy. Use `"cuda"` for GPUDirect RDMA
3. **Insufficient bandwidth** — KV cache size scales with `num_layers × num_heads × head_dim × seq_len × 2 (K+V) × dtype_bytes`. For Llama-70B with 8K context in bf16: ~5 GB per request. Requires 100+ Gbps network
4. **Side channel host misconfigured** — if cross-node, `VLLM_NIXL_SIDE_CHANNEL_HOST` must be set to the externally reachable address

### Load Imbalance

**Symptom**: Prefill instances idle while decode instances are overloaded (or vice versa).

Fixes:
1. **Adjust replica ratio** — typical: 1 prefill : 2-4 decode instances for chat workloads (short prompts, long generations)
2. **Monitor metrics**: `vllm:num_requests_running` on each instance type. Prefill should show short bursts; decode should show steady load
3. **Use `kv_role="kv_both"`** — allows instances to handle both phases, with the router deciding scheduling
4. For Ray Serve: use `autoscaling_config` with different `min_replicas`/`max_replicas` for prefill vs decode

### Model Mismatch

**Symptom**: `ValueError` or garbled output after KV transfer.

Both prefill and decode instances must use:
- Same model and revision
- Same `dtype`
- Same `block_size` (KV cache block size)
- Same `max_model_len` (or compatible)

### kv_load_failure_policy

When KV transfer fails mid-request:
- `"recompute"` (default) — decode instance recomputes prefill locally. Slower but resilient.
- `"fail"` — request fails immediately. Use only when you need strict PD enforcement.

## Limitations

- Both instances must load the same model (no asymmetric model pairing)
- KV transfer adds network latency — only beneficial when prefill is a bottleneck
- Requires high-bandwidth interconnect for production (RDMA/InfiniBand preferred, 100+ Gbps)
- `P2pNcclConnector` only supports exactly 2 instances (1 prefill + 1 decode)
- API is experimental — `kv_transfer_params` format may change
