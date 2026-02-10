---
name: ray-serve
description: >
  Scalable model serving with Ray Serve on Kubernetes. Use when: (1) Configuring ServeConfigV2
  (applications, deployments, proxy, HTTP/gRPC options), (2) Tuning autoscaling parameters
  (target_ongoing_requests, upscale/downscale delays, smoothing factors), (3) Configuring
  deployment resources (ray_actor_options, num_replicas, max_ongoing_requests),
  (4) Setting up request batching (@serve.batch parameters), (5) Composing multi-model
  pipelines with DeploymentHandle, (6) Deploying on Kubernetes with RayService CRD,
  (7) Configuring health checks, graceful shutdown, and logging,
  (8) Tuning performance (streaming, gRPC, multiplexed models),
  (9) Disaggregated prefill-decode serving with build_pd_openai_app.
---

# Ray Serve

Ray Serve is a scalable model serving framework built on Ray. Deploys on Kubernetes via the RayService CRD. Version: **2.53.0+**.

## ServeConfigV2 — Full Schema

The Serve config file is the production deployment format. It's embedded in the RayService CRD's `serveConfigV2` field.

```yaml
proxy_location: EveryNode          # Where to run HTTP/gRPC proxies

http_options:                      # Global HTTP settings (immutable at runtime)
  host: 0.0.0.0
  port: 8000
  request_timeout_s: 300
  keep_alive_timeout_s: 5

grpc_options:                      # Global gRPC settings (immutable at runtime)
  port: 9000
  grpc_servicer_functions:
    - my_module:add_MyServiceServicer_to_server
  request_timeout_s: 300

logging_config:                    # Global logging (overridable per-app/deployment)
  log_level: INFO
  logs_dir: null
  encoding: TEXT                   # TEXT or JSON
  enable_access_log: true

applications:                      # One or more applications
  - name: my-app
    route_prefix: /
    import_path: my_module:app
    runtime_env:
      pip: [torch, transformers]
      env_vars:
        MODEL_ID: meta-llama/Llama-3.1-8B
    args: {}                       # Passed to app builder function
    external_scaler_enabled: false # Enable external scaling REST API
    deployments:
      - name: MyDeployment
        # ... deployment settings (see below)
```

### Proxy Location

| Value | Behavior |
|---|---|
| `EveryNode` (default) | Run proxy on every node with at least one replica |
| `HeadOnly` | Single proxy on head node only |
| `Disabled` | No proxies — use DeploymentHandle only |

### HTTP Options

| Setting | Purpose | Default |
|---|---|---|
| `host` | Bind address | `0.0.0.0` |
| `port` | HTTP port | `8000` |
| `request_timeout_s` | End-to-end request timeout | None (no timeout) |
| `keep_alive_timeout_s` | HTTP keep-alive timeout | `5` |

### gRPC Options

| Setting | Purpose | Default |
|---|---|---|
| `port` | gRPC port | `9000` |
| `grpc_servicer_functions` | Import paths for gRPC servicer registration functions | `[]` |
| `request_timeout_s` | End-to-end request timeout | None |

### Logging Config

| Setting | Purpose | Default |
|---|---|---|
| `log_level` | Log level (DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `logs_dir` | Custom log directory | None |
| `encoding` | Log format: `TEXT` or `JSON` | `TEXT` |
| `enable_access_log` | Log every request | `true` |

Can be set globally, per-application, or per-deployment (most specific wins).

### Application Settings

| Setting | Purpose | Default |
|---|---|---|
| `name` | Unique application name | required |
| `route_prefix` | HTTP route prefix (must be unique) | `/` |
| `import_path` | Python import path to the Serve app | required |
| `runtime_env` | Runtime environment (pip packages, env vars, working_dir) | `{}` |
| `args` | Arguments passed to app builder function | `{}` |
| `external_scaler_enabled` | Enable external scaling REST API | `false` |
| `deployments` | Per-deployment overrides | `[]` |

## Deployment Settings

Every `@serve.deployment` accepts these settings. Set them in code (decorator or `.options()`) or override in the config file. **Config file takes highest priority**, then code, then defaults.

### Core Deployment Settings

| Setting | Purpose | Default |
|---|---|---|
| `name` | Deployment name (must match code) | Class/function name |
| `num_replicas` | Fixed replica count, or `"auto"` for autoscaling | `1` |
| `max_ongoing_requests` | Max concurrent requests per replica | `5` |
| `max_queued_requests` | Max queued requests per caller (experimental) | `-1` (no limit) |
| `user_config` | JSON-serializable config passed to `reconfigure()` | None |
| `logging_config` | Per-deployment logging override | Global config |

### Ray Actor Options

Resources allocated to each replica actor:

| Setting | Purpose | Default |
|---|---|---|
| `num_cpus` | CPU cores per replica | `1` |
| `num_gpus` | GPUs per replica | `0` |
| `memory` | Memory in bytes per replica | Auto |
| `accelerator_type` | Required accelerator (e.g., `A100`, `H100`) | None |
| `resources` | Custom resource labels | `{}` |
| `runtime_env` | Per-replica runtime environment | `{}` |
| `object_store_memory` | Object store memory per replica | Auto |

```yaml
deployments:
  - name: LLMDeployment
    num_replicas: 2
    ray_actor_options:
      num_cpus: 4
      num_gpus: 1
      accelerator_type: A100
      memory: 34359738368  # 32 GiB in bytes
```

> **Important:** `ray_actor_options` is treated as a single dict. Setting it in the config file completely replaces any values from code (not merged). Same applies to `user_config` and `autoscaling_config`.

### Health Check Settings

| Setting | Purpose | Default |
|---|---|---|
| `health_check_period_s` | Interval between health checks | `10` |
| `health_check_timeout_s` | Timeout for each health check | `30` |

Implement a custom health check:
```python
@serve.deployment
class MyModel:
    def check_health(self):
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
```

### Graceful Shutdown Settings

| Setting | Purpose | Default |
|---|---|---|
| `graceful_shutdown_wait_loop_s` | Wait interval checking for remaining work | `2` |
| `graceful_shutdown_timeout_s` | Max time before force kill | `20` |

## Autoscaling Configuration

Set `num_replicas: "auto"` or provide explicit `autoscaling_config`:

### autoscaling_config Settings

| Setting | Purpose | Default | Default with `auto` |
|---|---|---|---|
| `min_replicas` | Minimum replicas | `1` | `1` |
| `max_replicas` | Maximum replicas | `1` | `100` |
| `initial_replicas` | Starting replica count | None (uses `min_replicas`) | None |
| `target_ongoing_requests` | Target concurrent requests per replica | `2` | `2` |
| `metrics_interval_s` | How often replicas report metrics | `10.0` | `10.0` |
| `look_back_period_s` | Window for averaging metrics | `30.0` | `30.0` |
| `smoothing_factor` | Gain factor for scaling decisions | `1.0` | `1.0` |
| `upscale_smoothing_factor` | Override smoothing for upscaling | None (uses `smoothing_factor`) | None |
| `downscale_smoothing_factor` | Override smoothing for downscaling | None (uses `smoothing_factor`) | None |
| `upscaling_factor` | Multiplicative factor for upscale steps | None | None |
| `downscaling_factor` | Multiplicative factor for downscale steps | None | None |
| `upscale_delay_s` | Seconds to wait before upscaling | `30.0` | `30.0` |
| `downscale_delay_s` | Seconds to wait before downscaling | `600.0` | `600.0` |
| `downscale_to_zero_delay_s` | Extra delay before scaling to 0 | None | None |
| `aggregation_function` | How to aggregate metrics (`MEAN`, `MAX`) | `MEAN` | `MEAN` |

```yaml
deployments:
  - name: LLMDeployment
    max_ongoing_requests: 10
    autoscaling_config:
      min_replicas: 1
      max_replicas: 8
      target_ongoing_requests: 3
      upscale_delay_s: 10
      downscale_delay_s: 300
      upscale_smoothing_factor: 2.0      # aggressive upscale
      downscale_smoothing_factor: 0.5    # conservative downscale
      metrics_interval_s: 5
      look_back_period_s: 15
```

**Tuning guidelines:**
- `target_ongoing_requests` — lower = lower latency, higher = higher throughput
- `upscale_delay_s` — lower for bursty traffic, higher for steady traffic
- `downscale_delay_s` — keep high (300-600s) to avoid thrashing
- `smoothing_factor` > 1 = more aggressive scaling, < 1 = more conservative
- `min_replicas: 0` — enables scale-to-zero (adds cold start latency)

## Request Batching (@serve.batch)

| Setting | Purpose | Default |
|---|---|---|
| `max_batch_size` | Max requests per batch | `10` |
| `batch_wait_timeout_s` | Max wait for a full batch | `0.01` |
| `max_concurrent_batches` | Max batches running concurrently | `1` |
| `batch_size_fn` | Custom function to compute batch size | None |

```python
@serve.deployment
class BatchModel:
    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.1, max_concurrent_batches=2)
    async def handle_batch(self, requests: list[str]) -> list[str]:
        # Process entire batch at once (e.g., batched GPU inference)
        return self.model.predict(requests)

    async def __call__(self, request):
        return await self.handle_batch(request.query_params["text"])
```

**Tuning:** Set `max_batch_size` to your model's optimal batch size. Set `batch_wait_timeout_s` low for latency-sensitive, higher for throughput-sensitive. Increase `max_concurrent_batches` if GPU can handle multiple batches.

## user_config (Dynamic Reconfiguration)

Update deployment behavior without restarting replicas:

```python
@serve.deployment
class Model:
    def reconfigure(self, config: dict):
        """Called on initial deploy and every config update."""
        self.model_path = config["model_path"]
        self.temperature = config.get("temperature", 1.0)
        self.model = load_model(self.model_path)
```

```yaml
deployments:
  - name: Model
    user_config:
      model_path: meta-llama/Llama-3.1-8B
      temperature: 0.7
```

Update `user_config` in the config file and re-apply — replicas call `reconfigure()` without restart. Useful for: model version swaps, A/B test weights, feature flags, hyperparameters.

## Model Composition (DeploymentHandle)

Chain multiple deployments in a pipeline:

```python
from ray.serve.handle import DeploymentHandle

@serve.deployment
class Preprocessor:
    async def __call__(self, text: str) -> list[int]:
        return tokenize(text)

@serve.deployment
class Model:
    async def __call__(self, tokens: list[int]) -> str:
        return self.model.generate(tokens)

@serve.deployment
class Pipeline:
    def __init__(self, preprocessor: DeploymentHandle, model: DeploymentHandle):
        self.preprocessor = preprocessor
        self.model = model

    async def __call__(self, request) -> str:
        tokens = await self.preprocessor.remote(request.query_params["text"])
        return await self.model.remote(tokens)

app = Pipeline.bind(Preprocessor.bind(), Model.bind())
```

## Streaming Responses

```python
from starlette.responses import StreamingResponse

@serve.deployment
class StreamModel:
    async def __call__(self, request):
        async def generate():
            for token in self.model.stream(request.query_params["prompt"]):
                yield token
        return StreamingResponse(generate(), media_type="text/plain")
```

## Multiplexed Models (Multi-LoRA)

Load multiple model variants on the same replica:

```python
@serve.deployment(num_replicas=2)
class MultiLoRAModel:
    def __init__(self):
        self.base_model = load_base_model()
        self.adapters = {}

    @serve.multiplexed(max_num_models_per_replica=10)
    async def get_model(self, model_id: str):
        if model_id not in self.adapters:
            self.adapters[model_id] = load_adapter(model_id)
        return self.adapters[model_id]

    async def __call__(self, request):
        model_id = serve.get_multiplexed_model_id()
        adapter = await self.get_model(model_id)
        return self.base_model.generate(request, adapter)
```

## Disaggregated Prefill-Decode Serving

Ray Serve provides native PD disaggregation via `build_pd_openai_app`, which manages separate prefill and decode vLLM instances with automatic KV cache transfer routing.

### Python API

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

### YAML Config (for `serve deploy` or RayService CRD)

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
        runtime_env:
          env_vars:
            VLLM_NIXL_SIDE_CHANNEL_PORT: "5601"
```

Key differences from `build_openai_app`:
- Uses `build_pd_openai_app` instead of `build_openai_app`
- Takes `prefill_config` and `decode_config` instead of `llm_configs` list
- Both configs must specify the **same model** — Ray handles routing between prefill and decode
- Autoscaling is independent: scale decode replicas higher for generation-heavy workloads
- NixlConnector handles KV cache transfer automatically between instances

See `assets/serve_pd_config.yaml` for a complete deployment template.

## Kubernetes Deployment (RayService)

The ServeConfigV2 is embedded in the RayService CRD:

```yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: llm-service
spec:
  serviceUnhealthySecondThreshold: 900    # time before marking service unhealthy
  deploymentUnhealthySecondThreshold: 300  # time before marking deployment unhealthy
  serveConfigV2: |
    applications:
      - name: llm
        route_prefix: /
        import_path: serve_app:app
        runtime_env:
          pip: [vllm, transformers]
        deployments:
          - name: VLLMDeployment
            num_replicas: 2
            max_ongoing_requests: 24
            ray_actor_options:
              num_cpus: 4
              num_gpus: 1
            autoscaling_config:
              min_replicas: 1
              max_replicas: 4
              target_ongoing_requests: 8
  rayClusterConfig:
    headGroupSpec:
      template:
        spec:
          containers:
            - name: ray-head
              resources:
                limits:
                  cpu: "4"
                  memory: 8Gi
    workerGroupSpecs:
      - groupName: gpu-workers
        replicas: 2
        minReplicas: 1
        maxReplicas: 4
        template:
          spec:
            containers:
              - name: ray-worker
                resources:
                  limits:
                    cpu: "8"
                    memory: 32Gi
                    nvidia.com/gpu: "1"
```

### RayService-Specific Settings

| Setting | Purpose | Default |
|---|---|---|
| `serviceUnhealthySecondThreshold` | Seconds before marking service unhealthy | `900` |
| `deploymentUnhealthySecondThreshold` | Seconds before marking deployment unhealthy | `300` |

### High Availability

For HA, set `max_replicas_per_node: 1` to spread replicas across nodes:

```yaml
deployments:
  - name: MyDeployment
    num_replicas: 3
    max_replicas_per_node: 1
```

## Priority of Settings

1. **Serve config file** (highest) — overrides everything
2. **Application code** (`@serve.deployment` decorator or `.options()`)
3. **Ray Serve defaults** (lowest)

`ray_actor_options`, `user_config`, and `autoscaling_config` are each replaced as whole dicts (not merged) when specified in the config file.

## Debugging

See `references/performance.md` for performance tuning and troubleshooting.

## Cross-References

- [ray-core](../ray-core/) — Ray actors powering Serve deployments
- [kuberay](../kuberay/) — Deploy Serve on Kubernetes via RayService CRD
- [vllm](../vllm/) — Serve vLLM models with Ray Serve

## Reference

- [Serve config files](https://docs.ray.io/en/latest/serve/production-guide/config.html)
- [Deployment configuration](https://docs.ray.io/en/latest/serve/configure-serve-deployment.html)
- [AutoscalingConfig API](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.config.AutoscalingConfig.html)
- [Advanced autoscaling](https://docs.ray.io/en/latest/serve/advanced-guides/advanced-autoscaling.html)
- [RayService on K8s](https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/rayservice.html)
- `references/performance.md` — performance tuning and troubleshooting
- `assets/serve_config.yaml` — ServeConfigV2 example with multi-model deployment and autoscaling
- `assets/serve_pd_config.yaml` — Disaggregated prefill-decode config with NixlConnector and independent scaling
