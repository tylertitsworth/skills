# LMCache — KV Cache Sharing and Offloading

[LMCache](https://github.com/LMCache/LMCache) is a KV cache management layer that integrates with vLLM to enable KV cache sharing across instances and CPU/disk offloading.

## Use Cases

1. **KV cache sharing** — multiple vLLM replicas share a central KV cache store, avoiding redundant prefill computation for the same prompts
2. **Disaggregated serving** — transfer KV caches from prefill to decode instances via LMCacheConnectorV1
3. **CPU offloading** — offload KV cache to CPU memory when GPU memory is full

## Architecture

```
vLLM Instance 1 ──┐
                   ├──► LMCache Server (shared KV store)
vLLM Instance 2 ──┘
```

## Configuration

### vLLM Integration (kwargs)

```python
from vllm import LLM
from vllm.config import KVTransferConfig

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    gpu_memory_utilization=0.8,
    max_model_len=8000,
    kv_transfer_config=KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    ),
)
```

### LMCache Config File

Set `LMCACHE_CONFIG_FILE` env var in the container spec:

```yaml
# lmcache_config.yaml
local_cpu:
  enabled: true
  buffer_size_gb: 20          # CPU offloading buffer

remote:
  enabled: true
  url: "lmcache-server:8080"  # shared cache server endpoint
  serde: "naive"              # serialization format
```

### Kubernetes Deployment

```yaml
# LMCache Server (shared KV store)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lmcache-server
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: cacheserver
        image: lmcache/vllm-openai:latest
        command: ["lmcache_server"]
        args: ["--port", "8080", "--serde", "naive"]
        resources:
          requests:
            cpu: "4"
            memory: "10Gi"
---
# vLLM instances with LMCache
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-with-lmcache
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: vllm
        image: lmcache/vllm-openai:latest
        env:
        - name: LMCACHE_CONFIG_FILE
          value: "/config/lmcache_config.yaml"
        - name: VLLM_USE_V1
          value: "1"
        args:
        - --model=meta-llama/Llama-3.1-8B-Instruct
        - --gpu-memory-utilization=0.8
        - --kv-transfer-config={"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}
```

### For Disaggregated PD Serving

```python
# Prefill instance
KVTransferConfig(
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_producer",
    kv_connector_extra_config={"nixl_backends": ["UCX"]},
)

# Decode instance
KVTransferConfig(
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_consumer",
    kv_connector_extra_config={"nixl_backends": ["UCX"]},
)
```

## Key Settings

| Setting | Purpose | Default |
|---|---|---|
| `local_cpu.enabled` | Enable CPU offloading | `false` |
| `local_cpu.buffer_size_gb` | CPU buffer size in GB | None |
| `remote.enabled` | Enable shared cache server | `false` |
| `remote.url` | Cache server endpoint | None |
| `remote.serde` | Serialization format | `"naive"` |

## Limitations

- Requires vLLM v1 engine (`VLLM_USE_V1=1`)
- Not compatible with chunked prefill
- Use `lmcache/vllm-openai` container image (not standard vLLM image)
- Cache server is a single point of failure — no built-in replication
