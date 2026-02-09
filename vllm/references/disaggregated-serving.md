# Disaggregated Prefill-Decode Serving

Disaggregated prefill-decode (PD) serving separates the prefill phase (processing input prompts) and the decode phase (generating tokens) onto different vLLM instances. This is **experimental** in vLLM.

## Why Disaggregated Serving?

1. **Independent TTFT and ITL tuning** — assign different parallelism strategies to prefill and decode instances without tradeoffs
2. **Controlled tail latency** — decode instances are never interrupted by prefill work, eliminating ITL spikes
3. **Hardware optimization** — prefill is compute-bound (benefits from larger TP), decode is memory-bound (benefits from more KV cache)

> **Note:** Disaggregated prefill does NOT improve throughput. It improves latency characteristics.

## Architecture

```
Client Request
     │
     ▼
┌──────────┐     KV Transfer      ┌──────────┐
│  Prefill │ ──────────────────►  │  Decode  │
│ Instance │   (via Connector)    │ Instance │
│ (GPU 0-3)│                      │ (GPU 4-7)│
└──────────┘                      └──────────┘
```

Two vLLM instances run simultaneously:
- **Prefill instance** (`kv_role="kv_producer"`) — processes prompts, produces KV caches
- **Decode instance** (`kv_role="kv_consumer"`) — receives KV caches, generates tokens

A **connector** transfers KV caches between instances.

## Connector Types

| Connector | Transport | Use Case |
|---|---|---|
| `NixlConnector` | NIXL (RDMA/UCX) | Production — fully async, highest performance |
| `LMCacheConnectorV1` | LMCache + NIXL | Production — with KV cache sharing/offloading |
| `P2pNcclConnector` | NCCL P2P | Same-node GPU transfer |
| `MooncakeConnector` | Mooncake | Alternative transport |
| `OffloadingConnector` | CPU memory | KV offloading to CPU |
| `MultiConnector` | Multiple | Chain multiple connectors |
| `ExampleConnector` | Shared storage | Development/testing only |

## Configuration

### KVTransferConfig (kwargs)

```python
from vllm import LLM
from vllm.config import KVTransferConfig

# Prefill instance
prefill_llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    kv_transfer_config=KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_producer",
        kv_buffer_device="cuda",
    ),
)

# Decode instance
decode_llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    kv_transfer_config=KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_consumer",
        kv_buffer_device="cuda",
    ),
)
```

### KVTransferConfig Fields

| Field | Purpose | Values |
|---|---|---|
| `kv_connector` | Connector implementation | See table above |
| `kv_role` | Instance role | `kv_producer`, `kv_consumer`, `kv_both` |
| `kv_buffer_device` | Buffer device for KV transfer | `cuda`, `cpu` |
| `kv_connector_extra_config` | Connector-specific config | Dict |

### NixlConnector Extra Config

```python
KVTransferConfig(
    kv_connector="NixlConnector",
    kv_role="kv_both",
    kv_buffer_device="cuda",
    kv_connector_extra_config={
        "backends": ["UCX", "GDS"],     # NIXL backends
    },
)
```

### OffloadingConnector

Offload KV cache to CPU memory:

```python
KVTransferConfig(
    kv_connector="OffloadingConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "block_size": 64,               # tokens per block
        "cpu_bytes_to_use": 10_000_000_000,  # 10 GB CPU memory
    },
)
```

### MultiConnector (chaining)

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

## Kubernetes Deployment Pattern

Deploy as two separate Deployments with a shared Service/router:

```yaml
# Prefill Deployment — compute-optimized (higher TP)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-prefill
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: vllm
        args:
        - --model=meta-llama/Llama-3.1-70B-Instruct
        - --tensor-parallel-size=4
        - --kv-transfer-config={"kv_connector":"NixlConnector","kv_role":"kv_producer"}
        resources:
          limits:
            nvidia.com/gpu: "4"
---
# Decode Deployment — memory-optimized (more KV cache)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-decode
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: vllm
        args:
        - --model=meta-llama/Llama-3.1-70B-Instruct
        - --tensor-parallel-size=2
        - --gpu-memory-utilization=0.95
        - --kv-transfer-config={"kv_connector":"NixlConnector","kv_role":"kv_consumer"}
        resources:
          limits:
            nvidia.com/gpu: "2"
```

## Limitations

- Experimental — API may change
- Not compatible with chunked prefill (when using LMCache)
- Both instances must use the same model and tokenizer
- Requires high-bandwidth interconnect between instances for production use (RDMA/NVLink preferred)
