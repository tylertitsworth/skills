---
name: flyte-kuberay
description: >
  Run Ray workloads as Flyte tasks using the flytekitplugins-ray integration. Use when:
  (1) Configuring RayJobConfig (head node, worker groups, autoscaling, runtime env),
  (2) Setting per-node resources (CPU, GPU, memory) for Ray head and worker pods,
  (3) Understanding the execution flow (Flyte task → RayJob CRD → KubeRay operator → Ray cluster),
  (4) Using Ray Core, Ray Data, Ray Train, or Ray Serve within Flyte workflows,
  (5) Configuring worker group scaling (replicas, min/max, autoscaling),
  (6) Using PodTemplate for advanced pod customization (tolerations, node selectors, volumes),
  (7) Combining with @dynamic for runtime-determined Ray cluster sizes,
  (8) Troubleshooting Ray task failures in Flyte (cluster creation, job submission, resource issues).
---

# Flyte × KubeRay Integration

The `flytekitplugins-ray` package lets you run Ray workloads as Flyte tasks. Flyte creates an ephemeral Ray cluster (via the KubeRay operator) for each task execution, then submits your function as a Ray job.

**Requirements:**
- KubeRay operator installed in the K8s cluster
- `flytekitplugins-ray` in your task's container image
- Flyte's Ray backend plugin enabled (see [flyte-deployment](../flyte-deployment/) skill)

**Version compatibility:** `flyte >= 1.11.1` requires `kuberay >= 1.1.0`

## Execution Flow

Understanding how Flyte orchestrates Ray is essential for debugging:

```
@task(task_config=RayJobConfig(...))
         │
         ▼
Flytekit serializes RayJobConfig → TaskTemplate (Protobuf)
         │
         ▼
FlytePropeller sees task type "ray"
         │
         ▼
Ray backend plugin creates a RayJob CRD in K8s
         │
         ▼
KubeRay operator provisions Ray cluster (head + workers)
         │
         ▼
KubeRay submits your task function via Ray Job Submission API
         │
         ▼
ray.init() connects to the cluster automatically
         │
         ▼
Task function executes (Ray remote calls fan out to workers)
         │
         ▼
On completion: outputs serialized, cluster torn down (if shutdown_after_job_finishes=True)
```

**Key implications:**
- `ray.init()` is called automatically in `pre_execute` — don't call it yourself
- The `address` is set to the head node automatically
- Your task function runs on the Ray head node (as the driver)
- `@ray.remote` functions and actors are scheduled on worker nodes
- The cluster is ephemeral — it's created and destroyed per task execution

## RayJobConfig

### RayJobConfig Settings

| Setting | Purpose | Default |
|---|---|---|
| `worker_node_config` | List of worker group configs | required |
| `head_node_config` | Head node configuration | None (uses defaults) |
| `enable_autoscaling` | Enable Ray autoscaler | `False` |
| `runtime_env` | Ray runtime environment (pip packages, env vars, working dir) | None |
| `address` | Ray cluster address (auto-set, rarely override) | None |
| `shutdown_after_job_finishes` | Tear down cluster after task completes | `False` |
| `ttl_seconds_after_finished` | Seconds before cleaning up a finished cluster | None |

### HeadNodeConfig Settings

| Setting | Purpose | Default |
|---|---|---|
| `ray_start_params` | Parameters passed to `ray start` on head | None |
| `requests` | Resource requests (CPU, memory, GPU) | None (uses task-level) |
| `limits` | Resource limits | None |
| `pod_template` | Custom PodTemplate for advanced pod config | None |

### WorkerNodeConfig Settings

| Setting | Purpose | Default |
|---|---|---|
| `group_name` | Worker group name (must be unique per group) | required |
| `replicas` | Number of worker pods | `1` |
| `min_replicas` | Min replicas for autoscaling | None |
| `max_replicas` | Max replicas for autoscaling | None |
| `ray_start_params` | Parameters passed to `ray start` on workers | None |
| `requests` | Resource requests per worker | None |
| `limits` | Resource limits per worker | None |
| `pod_template` | Custom PodTemplate for workers | None |

## Resource Configuration

Resources can be set at three levels (most specific wins):

1. **Task-level** `requests`/`limits` in `@task()` — applies to ALL pods (head + workers)
2. **HeadNodeConfig** `requests`/`limits` — head pod only
3. **WorkerNodeConfig** `requests`/`limits` — per worker group

### GPU Training Example

```python
from flytekit import ImageSpec, Resources, task, workflow
from flytekitplugins.ray import RayJobConfig, HeadNodeConfig, WorkerNodeConfig

training_image = ImageSpec(
    name="ray-training",
    packages=["flytekitplugins-ray", "ray[default,train]", "torch", "transformers"],
    apt_packages=["wget"],  # kuberay readiness probe
    registry="ghcr.io/my-org",
)

ray_config = RayJobConfig(
    head_node_config=HeadNodeConfig(
        ray_start_params={"num-cpus": "0"},  # don't schedule work on head
        requests=Resources(cpu="4", mem="16Gi"),
    ),
    worker_node_config=[
        WorkerNodeConfig(
            group_name="gpu-workers",
            replicas=4,
            requests=Resources(cpu="8", mem="32Gi", gpu="1"),
            limits=Resources(gpu="1"),
        ),
    ],
    runtime_env={"env_vars": {"WANDB_PROJECT": "my-project"}},
    shutdown_after_job_finishes=True,
    ttl_seconds_after_finished=300,
)

@task(
    task_config=ray_config,
    container_image=training_image,
)
def distributed_training(model_name: str, epochs: int) -> float:
    import ray.train
    from ray.train.torch import TorchTrainer
    from ray.train import ScalingConfig

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
        train_loop_config={"model_name": model_name, "epochs": epochs},
    )
    result = trainer.fit()
    return result.metrics["eval_loss"]
```

### Multiple Worker Groups (Heterogeneous)

```python
ray_config = RayJobConfig(
    head_node_config=HeadNodeConfig(
        requests=Resources(cpu="4", mem="8Gi"),
    ),
    worker_node_config=[
        # GPU workers for training
        WorkerNodeConfig(
            group_name="gpu-workers",
            replicas=4,
            requests=Resources(cpu="8", mem="32Gi", gpu="1"),
        ),
        # CPU workers for data preprocessing
        WorkerNodeConfig(
            group_name="cpu-workers",
            replicas=8,
            requests=Resources(cpu="4", mem="16Gi"),
        ),
    ],
    shutdown_after_job_finishes=True,
)
```

### Autoscaling

```python
ray_config = RayJobConfig(
    worker_node_config=[
        WorkerNodeConfig(
            group_name="gpu-workers",
            replicas=2,       # initial replicas
            min_replicas=1,   # scale down to 1
            max_replicas=8,   # scale up to 8
            requests=Resources(cpu="8", mem="32Gi", gpu="1"),
        ),
    ],
    enable_autoscaling=True,
    shutdown_after_job_finishes=True,
)
```

## PodTemplate for Advanced Configuration

Use `PodTemplate` for node selectors, tolerations, volumes, and other pod-level settings:

```python
from flytekit import PodTemplate
from kubernetes.client import (
    V1PodSpec, V1Container, V1Volume, V1VolumeMount,
    V1EmptyDirVolumeSource, V1Toleration,
)

gpu_pod_template = PodTemplate(
    pod_spec=V1PodSpec(
        containers=[
            V1Container(
                name="ray-worker",
                volume_mounts=[
                    V1VolumeMount(name="shm", mount_path="/dev/shm"),
                    V1VolumeMount(name="data", mount_path="/data"),
                ],
            ),
        ],
        volumes=[
            V1Volume(name="shm", empty_dir=V1EmptyDirVolumeSource(
                medium="Memory", size_limit="16Gi"
            )),
            V1Volume(name="data", persistent_volume_claim={
                "claimName": "training-data"
            }),
        ],
        tolerations=[
            V1Toleration(key="nvidia.com/gpu", operator="Exists", effect="NoSchedule"),
        ],
        node_selector={"nvidia.com/gpu.product": "NVIDIA-A100-SXM4-80GB"},
    ),
)

ray_config = RayJobConfig(
    worker_node_config=[
        WorkerNodeConfig(
            group_name="gpu-workers",
            replicas=4,
            requests=Resources(cpu="8", mem="32Gi", gpu="1"),
            pod_template=gpu_pod_template,
        ),
    ],
    shutdown_after_job_finishes=True,
)
```

## Dynamic Cluster Sizing

Use `@dynamic` to determine cluster size at runtime:

```python
from flytekit import dynamic

@dynamic(container_image=training_image)
def adaptive_training(model_name: str, dataset_size: int) -> float:
    # Determine GPU count based on dataset size
    num_gpus = 4 if dataset_size > 1_000_000 else 2

    ray_config = RayJobConfig(
        worker_node_config=[
            WorkerNodeConfig(
                group_name="gpu-workers",
                replicas=num_gpus,
                requests=Resources(cpu="8", mem="32Gi", gpu="1"),
            ),
        ],
        shutdown_after_job_finishes=True,
    )

    return train_task(model_name=model_name).with_overrides(
        task_config=ray_config,
        requests=Resources(cpu="4", mem="8Gi"),
    )
```

## Common Patterns

### Ray Data Processing Pipeline

```python
@task(
    task_config=RayJobConfig(
        worker_node_config=[
            WorkerNodeConfig(group_name="data-workers", replicas=16,
                             requests=Resources(cpu="4", mem="16Gi")),
        ],
        shutdown_after_job_finishes=True,
    ),
    container_image=data_image,
)
def preprocess_dataset(input_path: str) -> str:
    import ray
    ds = ray.data.read_parquet(input_path)
    ds = ds.map(tokenize_fn).repartition(100)
    output_path = "/data/processed"
    ds.write_parquet(output_path)
    return output_path
```

### Ray Serve Deployment Validation

```python
@task(
    task_config=RayJobConfig(
        worker_node_config=[
            WorkerNodeConfig(group_name="serve-workers", replicas=2,
                             requests=Resources(cpu="4", mem="16Gi", gpu="1")),
        ],
        shutdown_after_job_finishes=True,
    ),
    container_image=serve_image,
)
def validate_serving(model_path: str) -> dict:
    import ray.serve
    # Deploy and test model serving, return latency/throughput metrics
    ...
```

### Full Training Pipeline

```python
@workflow
def training_pipeline(
    raw_data: str, model_name: str, epochs: int
) -> float:
    processed = preprocess_dataset(input_path=raw_data)     # Ray Data task
    metrics = distributed_training(                          # Ray Train task
        model_name=model_name, epochs=epochs
    )
    return metrics
```

## Runtime Environment

The `runtime_env` field installs dependencies at Ray cluster startup:

```python
RayJobConfig(
    runtime_env={
        "pip": ["numpy==1.26.0", "pandas"],
        "env_vars": {"TOKENIZERS_PARALLELISM": "false"},
        "working_dir": "./src",  # uploaded to cluster
    },
    ...
)

# Or from a requirements file:
RayJobConfig(
    runtime_env="./requirements.txt",
    ...
)
```

> **Prefer ImageSpec over runtime_env** for heavy dependencies (PyTorch, transformers). `runtime_env` is best for lightweight, frequently-changing deps. Heavy pip installs at cluster startup add significant latency.

## Flyte Backend Plugin Configuration

The Flyte admin must enable the Ray plugin. In Flyte's Helm values:

```yaml
# flyte-binary or flyte-core values
configmap:
  enabled_plugins:
    tasks:
      task-plugins:
        enabled-plugins:
          - container
          - ray
        default-for-task-types:
          container: container
          ray: ray
  ray:
    - address: ""
    - shutdown_after_job_finishes: true
```

## Troubleshooting

### Cluster Not Creating

| Symptom | Cause | Fix |
|---|---|---|
| Task stuck in `QUEUED` | KubeRay operator not installed | Install KubeRay operator in cluster |
| RayJob CRD not found | API version mismatch | Ensure `flyte >= 1.11.1` with `kuberay >= 1.1.0` |
| Pods pending | Insufficient cluster resources | Check node resources, adjust requests |
| GPU pods pending | No GPU nodes / wrong resource key | Verify `nvidia.com/gpu` resource, check node labels |

### Task Failures

| Symptom | Cause | Fix |
|---|---|---|
| `ray.init()` connection refused | Head node not ready | Increase `ttl_seconds_after_finished`, check head pod logs |
| Worker OOM killed | Memory limits too low | Increase worker `requests.mem` / `limits.mem` |
| `ModuleNotFoundError` | Missing from image or runtime_env | Add to ImageSpec `packages` or `runtime_env.pip` |
| Ingress webhook errors | Missing ingress controller | Install nginx-ingress or disable Ray dashboard ingress |
| Task timeout | Cluster creation too slow | Pre-pull images, increase Flyte task timeout |

### Debugging

```bash
# Check RayJob CRDs created by Flyte
kubectl get rayjobs -n <flyte-execution-ns>

# Check Ray cluster status
kubectl get rayclusters -n <flyte-execution-ns>

# Check head/worker pod status
kubectl get pods -n <flyte-execution-ns> -l ray.io/cluster=<cluster-name>

# View Ray head logs (driver output)
kubectl logs -n <flyte-execution-ns> <head-pod-name>

# Check KubeRay operator logs
kubectl logs -n ray-system deploy/kuberay-operator
```

## Cross-References

- [flyte-sdk](../flyte-sdk/) — Flytekit task/workflow authoring, ImageSpec, type system
- [flyte-deployment](../flyte-deployment/) — Flyte cluster setup and plugin configuration
- [kuberay](../kuberay/) — KubeRay operator CRDs, RayJob spec, troubleshooting
- [ray-train](../ray-train/) — Ray Train distributed training config
- [ray-core](../ray-core/) — Ray remote functions, actors, object store

## Reference

- [Flyte Ray plugin docs](https://docs-legacy.flyte.org/en/latest/flytesnacks/examples/ray_plugin/ray_example.html)
- [flytekitplugins-ray API](https://docs.flyte.org/en/latest/api/flytekit/plugins/generated/flytekitplugins.ray.RayJobConfig.html)
- [KubeRay RayJob spec](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/rayjob-quick-start.html)
