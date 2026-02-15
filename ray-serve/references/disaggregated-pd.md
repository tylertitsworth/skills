# Disaggregated Prefill-Decode Serving

Ray Serve provides native PD disaggregation via `build_pd_openai_app`, which manages separate prefill and decode vLLM instances with automatic KV cache transfer routing.

## Python API

```python
from ray import serve
from ray.serve.llm import LLMConfig, build_pd_openai_app

prefill_config = LLMConfig(
    model_loading_config={"model_id": "llama-8b", "model_source": "meta-llama/Llama-3.1-8B-Instruct"},
    accelerator_type="A100",
    deployment_config={"autoscaling_config": {"min_replicas": 1, "max_replicas": 2}},
    engine_kwargs={
        "tensor_parallel_size": 2,
        "kv_transfer_config": {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
    },
)

decode_config = LLMConfig(
    model_loading_config={"model_id": "llama-8b", "model_source": "meta-llama/Llama-3.1-8B-Instruct"},
    accelerator_type="A100",
    deployment_config={"autoscaling_config": {"min_replicas": 2, "max_replicas": 8}},
    engine_kwargs={
        "gpu_memory_utilization": 0.95,
        "kv_transfer_config": {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
    },
)

app = build_pd_openai_app({"prefill_config": prefill_config, "decode_config": decode_config})
serve.run(app)
```

## YAML Config (for `serve deploy` or RayService CRD)

```yaml
applications:
  - name: llm_pd_app
    route_prefix: /
    import_path: ray.serve.llm:build_pd_openai_app
    args:
      prefill_config:
        model_loading_config:
          model_id: llama-8b
          model_source: meta-llama/Llama-3.1-8B-Instruct
        accelerator_type: A100
        deployment_config:
          autoscaling_config:
            min_replicas: 1
            max_replicas: 2
        engine_kwargs:
          tensor_parallel_size: 2
          kv_transfer_config:
            kv_connector: NixlConnector
            kv_role: kv_both
      decode_config:
        model_loading_config:
          model_id: llama-8b
          model_source: meta-llama/Llama-3.1-8B-Instruct
        accelerator_type: A100
        deployment_config:
          autoscaling_config:
            min_replicas: 2
            max_replicas: 8
        engine_kwargs:
          gpu_memory_utilization: 0.95
          kv_transfer_config:
            kv_connector: NixlConnector
            kv_role: kv_both
```

## How PD Routing Works

`build_pd_openai_app` creates a `PDProxyServer` that orchestrates the request flow:

1. Client sends request to the OpenAI-compatible endpoint
2. PDProxyServer routes to a **prefill** replica — processes the prompt, transfers KV cache via NixlConnector
3. PDProxyServer routes to a **decode** replica with KV cache reference — generates tokens
4. Decode replica streams response back through the proxy to the client

For advanced control, use the component API directly:

```python
from ray import serve
from ray.serve.llm import LLMConfig, build_dp_deployment
from ray.serve.llm.deployment import PDProxyServer
from ray.serve.llm.ingress import OpenAiIngress, make_fastapi_ingress

prefill_deployment = build_dp_deployment(prefill_config, name_prefix="Prefill:")
decode_deployment = build_dp_deployment(decode_config, name_prefix="Decode:")

proxy_options = PDProxyServer.get_deployment_options(prefill_config, decode_config)
proxy = serve.deployment(PDProxyServer).options(**proxy_options).bind(
    prefill_server=prefill_deployment,
    decode_server=decode_deployment,
)

ingress_options = OpenAiIngress.get_deployment_options([prefill_config, decode_config])
ingress_cls = make_fastapi_ingress(OpenAiIngress)
app = serve.deployment(ingress_cls).options(**ingress_options).bind(llm_deployments=[proxy])
```

## LLMConfig Fields for PD

Both `prefill_config` and `decode_config` are `LLMConfig` objects. Key fields:

| Field | Type | Purpose |
|---|---|---|
| `model_loading_config` | dict | `model_id` (API name) and `model_source` (HF repo or path). **Must match between prefill and decode.** |
| `accelerator_type` | str | GPU type: `"A10G"`, `"L4"`, `"A100"`, `"H100"`. Used for scheduling. |
| `engine_kwargs` | dict | All vLLM kwargs (TP, memory, KV transfer config, etc.). TP is auto-managed but can be overridden. |
| `deployment_config` | dict | Ray Serve deployment options: `autoscaling_config`, `max_ongoing_requests`, `num_replicas`, `health_check_*`, `graceful_shutdown_*` |
| `placement_group_config` | dict | Custom resource bundles + strategy (`"PACK"` default, `"STRICT_PACK"` for DP). |
| `runtime_env` | dict | Environment variables (`UCX_NET_DEVICES`, etc.) and pip packages. |
| `lora_config` | dict | LoRA adapter settings (same adapters must be on both prefill and decode). |
| `experimental_configs` | dict | `stream_batching_interval_ms` (batching window), `num_ingress_replicas`, `dp_size_per_node` (for DP+PD). |
| `log_engine_metrics` | bool | Enable vLLM Prometheus metrics via Ray (default: `True`). |

## engine_kwargs for PD

The `engine_kwargs` dict is passed directly to vLLM. Key settings for PD:

| Kwarg | Prefill | Decode | Notes |
|---|---|---|---|
| `tensor_parallel_size` | Higher (e.g., 4) | Lower (e.g., 2) | Prefill is compute-bound, decode is memory-bound |
| `gpu_memory_utilization` | 0.85-0.90 | 0.90-0.95 | Decode needs more KV cache headroom |
| `max_num_seqs` | Lower | Higher (512+) | Decode handles more concurrent sequences |
| `enable_chunked_prefill` | `True` | — | Overlaps prefill chunks with scheduling |
| `enable_prefix_caching` | Optional | `True` | Reuse KV across requests on decode side |
| `kv_cache_dtype` | — | `"fp8"` | Halves KV cache memory and transfer size on decode |
| `kv_transfer_config` | Required | Required | `{"kv_connector": "NixlConnector", "kv_role": "kv_both"}` |

## Autoscaling PD Independently

Prefill and decode have different scaling characteristics:

```python
# Prefill: fewer replicas, bursty compute
prefill_config = LLMConfig(
    ...,
    deployment_config={
        "autoscaling_config": {
            "min_replicas": 1,
            "max_replicas": 4,
            "target_ongoing_requests": 8,      # Lower — prefill completes fast
            "upscale_delay_s": 15,              # Scale up quickly for prompt bursts
            "downscale_delay_s": 300,
        },
        "max_ongoing_requests": 32,
    },
)

# Decode: more replicas, sustained generation
decode_config = LLMConfig(
    ...,
    deployment_config={
        "autoscaling_config": {
            "min_replicas": 2,
            "max_replicas": 8,
            "target_ongoing_requests": 16,     # Higher — decode streams over time
            "upscale_delay_s": 30,
            "downscale_delay_s": 600,          # Slower downscale — generation is long-lived
        },
        "max_ongoing_requests": 128,
    },
)
```

## Data Parallel + PD (DP+PD)

Combine data parallelism with PD disaggregation for maximum throughput:

```python
from ray.serve.llm import LLMConfig, build_dp_deployment

prefill_config = LLMConfig(
    model_loading_config={"model_id": "llama-8b", "model_source": "meta-llama/Llama-3.1-8B-Instruct"},
    engine_kwargs={
        "data_parallel_size": 2,
        "tensor_parallel_size": 1,
        "kv_transfer_config": {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
    },
    experimental_configs={"dp_size_per_node": 2},
)

decode_config = LLMConfig(
    model_loading_config={"model_id": "llama-8b", "model_source": "meta-llama/Llama-3.1-8B-Instruct"},
    engine_kwargs={
        "data_parallel_size": 4,
        "tensor_parallel_size": 1,
        "kv_transfer_config": {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
    },
    experimental_configs={"dp_size_per_node": 4},
)
```

> **Note**: With DP+PD, `num_replicas` in `deployment_config` must match `data_parallel_size`. Autoscaling is **not supported** for DP deployments — use fixed replica counts.

## Constraints

- Both configs **must use the same model** and `model_id`
- `kv_transfer_config` is required in both configs' `engine_kwargs`
- NixlConnector is the recommended connector for Ray Serve PD
- `VLLM_NIXL_SIDE_CHANNEL_PORT` is managed automatically by Ray Serve for multi-node deployments — do not set it manually

See `assets/serve_pd_config.yaml` for a complete deployment template.
