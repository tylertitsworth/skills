---
name: ray-serve
description: >
  Deploy and serve ML models with Ray Serve — scalable online inference on Ray. Use when:
  (1) Deploying models as Serve deployments (single model, multi-model composition, DAGs),
  (2) Configuring autoscaling (replicas, target concurrency, scaling speed),
  (3) Setting up dynamic request batching for throughput optimization,
  (4) Building multi-model pipelines with DeploymentHandle composition,
  (5) Integrating with FastAPI for custom HTTP endpoints,
  (6) Streaming responses (LLM token-by-token output),
  (7) Deploying on Kubernetes via KubeRay RayService,
  (8) Debugging serving issues (latency, OOM, cold starts, queue backlogs),
  (9) Performance tuning (async methods, resource allocation, gRPC).
---

# Ray Serve

Scalable model serving framework built on Ray. Serves ML models and arbitrary Python logic with autoscaling, batching, composition, and streaming.

**Docs:** https://docs.ray.io/en/latest/serve/index.html
**Version:** Ray 2.53.0

## Basic Deployment

```python
from ray import serve
from starlette.requests import Request

@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    num_replicas=2,
)
class TextClassifier:
    def __init__(self):
        from transformers import pipeline
        self.model = pipeline("text-classification", device="cuda:0")

    async def __call__(self, request: Request):
        data = await request.json()
        return self.model(data["text"])[0]

app = TextClassifier.bind()

# Run locally
serve.run(app, route_prefix="/classify")

# Query
# curl -X POST http://localhost:8000/classify -d '{"text": "hello world"}'
```

### Deployment Options

```python
@serve.deployment(
    num_replicas=2,                           # fixed replica count
    # OR
    num_replicas="auto",                      # enable autoscaling with defaults
    max_ongoing_requests=5,                   # max concurrent requests per replica
    ray_actor_options={
        "num_gpus": 1,                        # GPU per replica
        "num_cpus": 2,                        # CPUs per replica
        "memory": 8 * 1024**3,                # memory reservation (bytes)
        "resources": {"TPU": 1},              # custom resources
    },
    health_check_period_s=10,                 # health check interval
    health_check_timeout_s=30,                # health check timeout
    graceful_shutdown_timeout_s=20,           # drain time on shutdown
)
class MyDeployment: ...
```

### Fractional GPUs

Share a GPU between multiple models:

```python
@serve.deployment(ray_actor_options={"num_gpus": 0.5})
class SmallModel: ...

@serve.deployment(ray_actor_options={"num_gpus": 0.5})
class AnotherSmallModel: ...
```

## Autoscaling

```python
@serve.deployment(
    num_replicas="auto",                     # enables autoscaling
    autoscaling_config={
        "target_ongoing_requests": 2,        # target concurrent requests per replica
        "min_replicas": 1,                   # minimum replicas (0 = scale to zero)
        "max_replicas": 20,                  # maximum replicas
        "upscale_delay_s": 3,                # wait before upscaling
        "downscale_delay_s": 300,            # wait before downscaling
        "initial_replicas": 2,               # replicas at startup
    },
    max_ongoing_requests=5,                  # backpressure limit per replica
)
class AutoscaledModel: ...
```

**Key tuning guidelines:**
- `target_ongoing_requests`: Lower = lower latency, higher = better throughput
- `max_ongoing_requests`: Set ~2-3× target for headroom
- `min_replicas=0`: Enables scale-to-zero (cold start on first request)
- `max_replicas`: Set ~20% higher than peak need

## Dynamic Request Batching

Batch individual requests for vectorized GPU inference:

```python
from typing import List

@serve.deployment(ray_actor_options={"num_gpus": 1})
class BatchedModel:
    def __init__(self):
        self.model = load_model()

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.1)
    async def __call__(self, inputs: List[str]) -> List[dict]:
        # inputs is a list of individual request values
        results = self.model.predict_batch(inputs)
        return results  # must return list of same length as inputs
```

### Batch Parameters

| Parameter | Default | Purpose |
|---|---|---|
| `max_batch_size` | 10 | Maximum batch size |
| `batch_wait_timeout_s` | 0.01 | Max wait for batch to fill |
| `max_concurrent_batches` | 1 | Parallel batch processing |
| `batch_size_fn` | `len` | Custom batch size metric (e.g., total tokens) |

### Custom Batch Size (e.g., Token-Based)

```python
def count_tokens(texts: List[str]) -> int:
    return sum(len(t.split()) for t in texts)

@serve.batch(max_batch_size=2048, batch_size_fn=count_tokens)
async def predict(self, texts: List[str]) -> List[dict]: ...
```

## Model Composition

Combine multiple deployments with `DeploymentHandle`:

```python
from ray.serve.handle import DeploymentHandle

@serve.deployment(ray_actor_options={"num_gpus": 1})
class Encoder:
    def encode(self, text: str):
        return self.model.encode(text)

@serve.deployment(ray_actor_options={"num_gpus": 1})
class Ranker:
    def rank(self, query_emb, doc_embs):
        return sorted_results(query_emb, doc_embs)

@serve.deployment
class SearchPipeline:
    def __init__(self, encoder: DeploymentHandle, ranker: DeploymentHandle):
        self.encoder = encoder
        self.ranker = ranker

    async def __call__(self, request: Request):
        data = await request.json()
        query_emb = await self.encoder.encode.remote(data["query"])
        doc_embs = [
            self.encoder.encode.remote(doc) for doc in data["documents"]
        ]
        return await self.ranker.rank.remote(query_emb, doc_embs)

# Bind the DAG
app = SearchPipeline.bind(
    Encoder.bind(),
    Ranker.bind(),
)
serve.run(app)
```

Each deployment in the composition scales independently.

## FastAPI Integration

```python
from fastapi import FastAPI

fastapi_app = FastAPI()

@serve.deployment(ray_actor_options={"num_gpus": 1})
@serve.ingress(fastapi_app)
class LLMService:
    def __init__(self):
        self.model = load_llm()

    @fastapi_app.post("/generate")
    async def generate(self, prompt: str, max_tokens: int = 256):
        return {"text": self.model.generate(prompt, max_tokens)}

    @fastapi_app.get("/health")
    async def health(self):
        return {"status": "ok"}

app = LLMService.bind()
```

## Streaming Responses

For LLM token-by-token streaming:

```python
from starlette.responses import StreamingResponse

@serve.deployment(ray_actor_options={"num_gpus": 1})
class StreamingLLM:
    def __init__(self):
        self.model = load_llm()

    async def __call__(self, request: Request):
        data = await request.json()

        async def token_generator():
            for token in self.model.generate_stream(data["prompt"]):
                yield token

        return StreamingResponse(token_generator(), media_type="text/plain")
```

Also works with `DeploymentHandle` streaming:

```python
# Caller side
handle = serve.get_deployment_handle("StreamingLLM")
response = handle.options(stream=True).remote(request)
async for token in response:
    print(token, end="")
```

## Kubernetes Deployment (RayService)

Deploy via KubeRay's RayService CRD. See the **kuberay** skill's `references/rayservice.md` for full details.

Quick reference:

```yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: llm-service
spec:
  serveConfigV2: |
    applications:
    - name: llm
      route_prefix: /
      import_path: serve_app:app
      deployments:
      - name: LLMService
        num_replicas: auto
        ray_actor_options:
          num_gpus: 1
  rayClusterConfig:
    rayVersion: "2.53.0"
    headGroupSpec: ...
    workerGroupSpecs: ...
```

## Key Commands

```bash
# Deploy from config file
serve deploy config.yaml

# Check status
serve status

# List applications
serve status --address http://localhost:52365

# Shutdown
serve shutdown
```

## Troubleshooting & Performance

For debugging latency, OOM, cold starts, and performance tuning, see `references/performance.md`.
