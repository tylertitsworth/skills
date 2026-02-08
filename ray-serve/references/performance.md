# Ray Serve Performance & Troubleshooting

## Table of Contents

- [Architecture overview](#architecture-overview)
- [Performance tuning](#performance-tuning)
- [Debugging latency](#debugging-latency)
- [Debugging OOM](#debugging-oom)
- [Cold starts](#cold-starts)
- [Monitoring](#monitoring)
- [Common issues](#common-issues)

## Architecture Overview

Request flow: Client → HTTP Proxy → Router → Replica (actor)

- **HTTP Proxy**: Accepts HTTP/gRPC requests, routes to deployments
- **Router**: Load-balances across replicas using power-of-2-choices
- **Replica**: Ray actor running your deployment code
- **Controller**: Manages deployments, autoscaling, health checks

Each replica processes requests concurrently up to `max_ongoing_requests`.

## Performance Tuning

### Use Async Methods

Async handlers allow concurrent request processing within a single replica:

```python
@serve.deployment(max_ongoing_requests=100)
class AsyncModel:
    async def __call__(self, request: Request):
        data = await request.json()
        result = await self.model.predict_async(data)
        return result
```

**Warning:** `def` (sync) handlers in FastAPI run in a thread pool, which can cause unexpected concurrency and OOM. Prefer `async def`.

### Throughput-Optimized Mode

For maximum throughput, set the environment variable:

```bash
RAY_SERVE_THROUGHPUT_OPTIMIZED=1
```

This disables thread isolation and reduces logging overhead. Requires all handler code to be non-blocking (use `await`, not `time.sleep`).

### OMP_NUM_THREADS

Deep learning frameworks use this for threading. Ray sets `OMP_NUM_THREADS=1` by default unless `num_cpus` is specified:

```python
@serve.deployment(ray_actor_options={"num_cpus": 4})  # OMP_NUM_THREADS=4
class Model: ...
```

### Request Timeout

Set a global timeout to prevent slow requests from blocking replicas:

```yaml
http_options:
  request_timeout_s: 30
```

Clients receive HTTP 408 on timeout and should retry.

### max_ongoing_requests vs target_ongoing_requests

| Parameter | Scope | Purpose |
|---|---|---|
| `max_ongoing_requests` | Per replica | Hard limit — rejects/queues beyond this |
| `target_ongoing_requests` | Autoscaler | Target average — triggers scaling decisions |

**Rule:** Set `max_ongoing_requests` ≈ 2-3× `target_ongoing_requests`.

If `max_ongoing_requests` is too low, requests queue at the router. If too high, replicas may OOM or have high latency.

### Batching + Autoscaling Together

When using `@serve.batch` with autoscaling, tune carefully:

```python
@serve.deployment(
    max_ongoing_requests=32,
    autoscaling_config={
        "target_ongoing_requests": 8,
        "min_replicas": 1,
        "max_replicas": 10,
    },
)
class BatchedModel:
    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.1)
    async def __call__(self, inputs: List[str]) -> List[str]:
        return self.model.batch_predict(inputs)
```

- `max_batch_size` should ≤ `max_ongoing_requests`
- Lower `batch_wait_timeout_s` = lower latency, smaller batches
- Higher `batch_wait_timeout_s` = better throughput, larger batches

## Debugging Latency

### Metrics to Watch

| Metric | Meaning |
|---|---|
| `serve_deployment_processing_latency_ms` | Time spent in your handler |
| `serve_num_router_requests_total` | Total requests routed |
| `serve_replica_processing_queries` | Current concurrent requests per replica |
| `serve_deployment_queued_queries` | Requests waiting for a replica |

### High Latency Checklist

1. **Queue building up?** → `serve_deployment_queued_queries` increasing → add replicas or increase `max_ongoing_requests`
2. **Processing time high?** → `serve_deployment_processing_latency_ms` → optimize your model or use batching
3. **Cold start delays?** → New replicas taking long to initialize → increase `min_replicas`, pre-warm models
4. **CPU contention?** → Check `OMP_NUM_THREADS`, set `num_cpus` appropriately
5. **Network bottleneck?** → Large request/response payloads → compress or stream

### Request Tracing

```python
# Add request ID header for tracing
import uuid

@serve.deployment
class Traced:
    async def __call__(self, request: Request):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        logger.info(f"Processing {request_id}")
        # ...
```

## Debugging OOM

### Common Causes

1. **Model too large for GPU** → Use quantization, model parallelism, or larger GPU
2. **Too many concurrent requests** → Lower `max_ongoing_requests`
3. **Sync FastAPI handlers** → Thread pool creates unexpected parallelism → use `async def`
4. **Memory leak in handler** → Profile with `tracemalloc`, check for accumulating state
5. **Large request/response objects** → Stream responses, compress payloads

### GPU OOM Specifically

```python
# Set per-process GPU memory fraction
import torch
torch.cuda.set_per_process_memory_fraction(0.9)

# Or use fractional GPUs to limit allocation
@serve.deployment(ray_actor_options={"num_gpus": 0.5})
class Model: ...
```

## Cold Starts

When `min_replicas=0` (scale-to-zero), the first request triggers replica creation:

1. Ray schedules the actor (~seconds)
2. `__init__` runs (model download + load — can be minutes for large models)
3. Request is processed

**Mitigation:**
- Set `min_replicas=1` for latency-sensitive services
- Pre-download model weights in the container image
- Use `initial_replicas` in autoscaling config for faster warmup
- Cache model weights in shared storage (NFS, S3)

## Monitoring

### Prometheus Metrics

Ray Serve exports metrics at `localhost:8080/metrics` on each node:

```yaml
# ServiceMonitor for Prometheus Operator
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ray-serve-monitor
spec:
  selector:
    matchLabels:
      ray.io/serve: "true"
  endpoints:
  - port: metrics
    path: /metrics
```

### Key Metrics

| Metric | Use |
|---|---|
| `serve_deployment_processing_latency_ms` | Handler latency per deployment |
| `serve_deployment_queued_queries` | Queue depth (scaling signal) |
| `serve_num_router_requests_total` | Total throughput |
| `serve_replica_processing_queries` | Per-replica concurrency |
| `serve_handle_request_counter` | Requests per DeploymentHandle |

### Grafana Dashboard

Ray provides a built-in Grafana dashboard. Import it from the Ray Dashboard (port 8265) → Metrics tab.

## Common Issues

| Issue | Symptom | Fix |
|---|---|---|
| Requests rejected | HTTP 503 | Increase `max_ongoing_requests` or add replicas |
| Timeout errors | HTTP 408 | Increase `request_timeout_s` or optimize handler |
| Replicas not scaling | Constant replica count | Check `max_replicas > 1`, verify autoscaling_config |
| Scale-to-zero not working | Replicas stay at 1 | Set `min_replicas: 0` in autoscaling_config |
| Model not using GPU | Slow inference on CPU | Set `ray_actor_options={"num_gpus": 1}` |
| High tail latency | P99 >> P50 | Enable batching, check for GC pauses, increase replicas |
| Deployment stuck updating | New version won't deploy | Check `serve status` for errors, check replica logs |
