---
name: ray-core
description: >
  Write distributed Python applications with Ray Core — tasks, actors, and the object store.
  Use when: (1) Writing Ray tasks (@ray.remote functions) with CPU/GPU/memory resource requirements,
  (2) Building Ray actors (stateful distributed objects) with concurrency and fault tolerance,
  (3) Using the object store (ray.put, ray.get, passing object references between tasks),
  (4) Implementing distributed patterns (map-reduce, pipeline parallelism, parameter server, tree-reduce),
  (5) Configuring fault tolerance (task retries, actor restarts, object reconstruction),
  (6) Performance tuning (scheduling strategies, placement groups, avoiding anti-patterns),
  (7) Debugging Ray applications (ray status, dashboard, timeline, common errors).
---

# Ray Core

Distributed computing primitives for Python. Three building blocks: **tasks**, **actors**, and **objects**.

**Docs:** https://docs.ray.io/en/latest/ray-core/walkthrough.html  
**Version:** Ray 2.53.0

## Initialization

```python
import ray

# Local mode
ray.init()

# Connect to existing cluster
ray.init(address="ray://head-svc:10001")  # via Ray client
ray.init(address="auto")                   # auto-detect (inside Ray pod)

# Configuration
ray.init(
    num_cpus=8,                    # override detected CPUs (local only)
    num_gpus=2,                    # override detected GPUs
    object_store_memory=10**10,    # 10GB object store
    runtime_env={                  # per-job dependencies
        "pip": ["torch==2.5.0"],
        "env_vars": {"KEY": "val"},
        "working_dir": "./src",
    },
    logging_level="INFO",
    dashboard_host="0.0.0.0",
)
```

### Runtime Environments

Install dependencies dynamically per-job, per-task, or per-actor:

```python
# Per-job (in ray.init)
ray.init(runtime_env={"pip": ["transformers"], "env_vars": {"HF_HOME": "/cache"}})

# Per-task override
@ray.remote(runtime_env={"pip": ["scipy"]})
def special_task(): ...

# Per-actor override
actor = MyActor.options(runtime_env={"pip": ["xgboost"]}).remote()
```

Fields: `pip`, `conda`, `env_vars`, `working_dir`, `py_modules`, `excludes`, `container`.

## Tasks

Stateless remote functions. Decorate with `@ray.remote`, call with `.remote()`:

```python
import ray

@ray.remote
def train_model(config):
    # runs on a remote worker
    return {"loss": 0.1, "config": config}

# Launch 4 tasks in parallel
futures = [train_model.remote({"lr": lr}) for lr in [0.1, 0.01, 0.001, 0.0001]]
results = ray.get(futures)  # blocks until all complete
```

### Resource Requirements

```python
# CPU only (default: 1 CPU)
@ray.remote(num_cpus=4)
def cpu_task(): ...

# GPU task
@ray.remote(num_gpus=1)
def gpu_task(): ...

# Multiple GPUs + memory
@ray.remote(num_gpus=2, memory=16 * 1024**3)  # memory in bytes
def large_gpu_task(): ...

# Custom resources
@ray.remote(resources={"TPU": 1})
def tpu_task(): ...

# Override at call time
gpu_task.options(num_gpus=2).remote()
```

### Task Options

```python
@ray.remote(
    num_cpus=1,
    num_gpus=0,
    memory=0,                    # bytes, 0 = no reservation
    max_retries=3,               # retries on system failure (-1 = infinite)
    retry_exceptions=[ValueError],  # also retry on these exceptions
    num_returns=2,               # multiple return values
    scheduling_strategy="SPREAD",
)
def my_task(): ...
```

## Actors

Stateful distributed objects. Each actor runs in its own process, methods execute serially:

```python
@ray.remote(num_gpus=1)
class ModelServer:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.request_count = 0

    def predict(self, batch):
        self.request_count += 1
        return self.model(batch)

    def get_stats(self):
        return {"requests": self.request_count}

# Create actor (starts a new process)
server = ModelServer.remote("/models/bert")

# Call methods (async, returns ObjectRef)
result = server.predict.remote(data)
print(ray.get(result))
```

### Actor Options

```python
@ray.remote(
    num_cpus=1,
    num_gpus=1,
    max_restarts=3,          # auto-restart on crash (-1 = infinite)
    max_task_retries=-1,     # retry methods on actor crash (-1 = infinite)
    max_concurrency=10,      # concurrent method execution (async actors)
    lifetime="detached",     # survives creator death, needs ray.kill()
    name="my_server",        # named actor, retrievable via ray.get_actor()
    namespace="prod",        # actor namespace
)
class MyActor: ...
```

### Async Actors

For I/O-bound workloads, use async methods with `max_concurrency`:

```python
@ray.remote(max_concurrency=10)
class AsyncWorker:
    async def fetch(self, url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                return await resp.text()
```

### Named and Detached Actors

```python
# Create a named, detached actor
counter = Counter.options(name="global_counter", lifetime="detached").remote()

# Retrieve from anywhere in the cluster
counter = ray.get_actor("global_counter")
ray.get(counter.get.remote())

# Cleanup
ray.kill(counter)
```

## Object Store

Ray's distributed shared-memory object store (Plasma) passes data between tasks/actors.

```python
# Explicitly put data into object store
large_data = ray.put(np.zeros((10000, 10000)))  # returns ObjectRef

# Pass refs to tasks (zero-copy on same node)
result = process.remote(large_data)  # no serialization if same node

# Get values back
value = ray.get(result)

# Wait for first N ready (non-blocking)
ready, not_ready = ray.wait(futures, num_returns=1, timeout=5.0)
```

### Key Rules

- Objects are **immutable** once in the store — numpy arrays become read-only (copy before writing)
- Small returns (≤100KB) inlined directly to caller — no store overhead
- Large objects in shared memory — zero-copy reads on same node
- Object refs can be passed as arguments (lazy execution, no intermediate fetch)
- Garbage collected when no refs remain
- Serialization: Pickle protocol 5 + cloudpickle for lambdas/nested functions
- NumPy arrays use out-of-band data (Pickle 5) — stored directly in shared memory

## Fault Tolerance

| Mechanism | Config | Behavior |
|---|---|---|
| Task retries | `max_retries=3` | Retry on system failure (node death, OOM) |
| Exception retries | `retry_exceptions=[Exc]` | Also retry on specified app exceptions |
| Actor restart | `max_restarts=3` | Reconstruct actor (reruns `__init__`) |
| Actor task retry | `max_task_retries=-1` | Retry pending methods after actor restart |
| Object reconstruction | Automatic | Re-execute task that created a lost object |

**At-least-once semantics:** Retried tasks/methods may execute more than once. Use for idempotent or read-only workloads, or implement checkpointing for stateful actors.

## Scheduling Strategies

```python
from ray.util.scheduling_strategies import (
    PlacementGroupSchedulingStrategy,
    NodeAffinitySchedulingStrategy,
)

# DEFAULT — hybrid locality + load balancing (recommended for most cases)
task.options(scheduling_strategy="DEFAULT").remote()

# SPREAD — distribute across nodes evenly
task.options(scheduling_strategy="SPREAD").remote()

# Node affinity — pin to specific node
task.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
    node_id=ray.get_runtime_context().get_node_id(),
    soft=False,
)).remote()
```

## Placement Groups

Reserve resources atomically across nodes (gang scheduling):

```python
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

# Reserve 4 GPU bundles, pack on fewest nodes
pg = placement_group([{"GPU": 1, "CPU": 4}] * 4, strategy="PACK")
ray.get(pg.ready())  # wait for resources

# Schedule on the placement group
@ray.remote(num_gpus=1)
def train_worker(rank): ...

futures = [
    train_worker.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_bundle_index=i
        )
    ).remote(i)
    for i in range(4)
]
```

| Strategy | Behavior |
|---|---|
| `PACK` | Pack bundles on fewest nodes (locality) |
| `SPREAD` | Spread bundles across nodes (fault tolerance) |
| `STRICT_PACK` | All bundles on one node (fails if impossible) |
| `STRICT_SPREAD` | Each bundle on different node (fails if impossible) |

## Cross-Language Support

Ray supports Java and C++ tasks/actors that interoperate with Python:

```python
# Call a Java actor from Python
java_actor = ray.cross_language.java_actor_class("com.example.MyActor").remote()
result = java_actor.method.remote(arg)
```

## Generator Tasks

Stream results incrementally from a task:

```python
@ray.remote(num_returns="dynamic")
def generate_data(n):
    for i in range(n):
        yield ray.put(i)  # yields ObjectRef per item

gen = generate_data.remote(100)
for ref in gen:
    print(ray.get(ref))
```

## Debugging

```bash
# Cluster resource status
ray status

# Dashboard (default port 8265)
# Expose dashboard via Ingress/Service on port 8265

# List tasks/actors
ray list tasks
ray list actors
ray summary actors

# Timeline profiling
ray timeline  # generates chrome://tracing compatible JSON
```

```python
# Inside a task/actor
ctx = ray.get_runtime_context()
ctx.get_node_id()
ctx.get_task_id()
ctx.get_actor_id()
ctx.get_job_id()
ctx.get_runtime_env()
```

### Common Errors

| Error | Cause | Fix |
|---|---|---|
| `RayOutOfMemoryError` | Object store full | Reduce object lifetimes, increase `object_store_memory` |
| `assignment destination is read-only` | Writing to object-store numpy array | `arr = arr.copy()` before mutation |
| `ObjectLostError` | Node holding object died | Enable task retries / object reconstruction |
| `ActorDiedError` | Actor process crashed | Set `max_restarts`, check OOM / exceptions |
| `ModuleNotFoundError` | Missing deps on worker | Use `runtime_env` with pip packages |

## Patterns and Anti-Patterns

For distributed computing patterns and common anti-patterns, see `references/patterns.md`.
