---
name: flyte-sdk
description: >
  Author ML workflows with Flytekit (Flyte Python SDK). Use when: (1) Writing tasks with
  @task decorator and configuring resources, images, retries, (2) Building ImageSpec for
  reproducible container environments, (3) Composing workflows with @workflow and @eager,
  (4) Using the type system (StructuredDataset, FlyteFile, FlyteDirectory, dataclasses),
  (5) Configuring dynamic workflows and map_task for fan-out, (6) Creating LaunchPlans with
  schedules and fixed inputs, (7) Understanding how workflows are serialized, registered,
  and executed on Kubernetes, (8) Debugging task execution failures.
---

# Flyte SDK (Flytekit)

Flytekit is the Python SDK for authoring Flyte workflows. Version: **1.16.x+**.

## How Flyte Executes Your Code

Understanding the execution model is essential for writing correct Flyte code.

### Compilation vs Execution

Flytekit operates in two modes:

1. **Compilation mode** — When you `pyflyte register` or `pyflyte package`, flytekit imports your module and traces the `@workflow` function to build a DAG. It does NOT execute task bodies. It serializes everything to Protobuf (TaskTemplate + WorkflowTemplate) and registers them with FlyteAdmin.

2. **Execution mode** — When FlytePropeller schedules a task, it creates a K8s Pod with your container image and runs `pyflyte-execute` as the entrypoint, which deserializes inputs, calls your task function, and serializes outputs.

**Implications:**
- Code at module level runs during BOTH compilation and execution
- Code inside `@task` bodies runs ONLY during execution (in a pod)
- `@workflow` bodies are traced for DAG structure, never executed directly — you can't use Python if/else or loops (use `conditional` or `@dynamic` instead)

### The Kubernetes Execution Path

```
pyflyte register → FlyteAdmin (stores Protobuf specs)
                         ↓
                   FlytePropeller (K8s operator, watches FlyteWorkflow CRDs)
                         ↓
                   Creates K8s Pod per task (or delegates to operator plugin)
                         ↓
                   Pod runs: pyflyte-execute --task-module my.module --task-name my_task
                         ↓
                   Task function executes, outputs written to blob store
                         ↓
                   FlytePropeller reads outputs, schedules downstream tasks
```

**Plugin types for task execution:**
- **Container task** (default): Creates a bare K8s Pod
- **K8s operator plugin**: Delegates to Spark, Ray, MPI, PyTorch operators
- **Agent plugin**: Calls external services (SageMaker, BigQuery, etc.)

## ImageSpec

ImageSpec defines the container image for tasks declaratively. Flytekit builds images automatically during registration.

### ImageSpec Settings

| Setting | Purpose | Example |
|---|---|---|
| `name` | Image name | `"my-training-image"` |
| `base_image` | Base Docker image | `"nvcr.io/nvidia/pytorch:24.01-py3"` |
| `packages` | pip packages | `["torch==2.5.0", "transformers"]` |
| `conda_packages` | Conda packages | `["cudatoolkit=12.1"]` |
| `conda_channels` | Conda channels | `["nvidia", "conda-forge"]` |
| `apt_packages` | apt-get packages | `["git", "curl"]` |
| `env` | Environment variables | `{"CUDA_HOME": "/usr/local/cuda"}` |
| `registry` | Container registry | `"ghcr.io/my-org"` |
| `python_version` | Python version | `"3.11"` |
| `pip_index` | Custom PyPI index | `"https://my-pypi.internal/simple"` |
| `pip_extra_index_url` | Extra index URLs | `["https://download.pytorch.org/whl/cu121"]` |
| `source_copy_mode` | Copy local source | `CopyFileDetection.ALL` |
| `commands` | Extra build commands | `["pip install flash-attn --no-build-isolation"]` |
| `builder` | Image builder | `"default"`, `"envd"`, `"noop"` |
| `platform` | Target platform | `"linux/amd64"` |
| `tag_format` | Tag format template | `"{spec_hash}-gpu"` |

### ImageSpec Examples

```python
from flytekit import ImageSpec, task
from flytekit.image_spec import CopyFileDetection

# GPU training image
training_image = ImageSpec(
    name="training",
    base_image="nvcr.io/nvidia/pytorch:24.01-py3",
    packages=[
        "transformers==4.46.0",
        "datasets==3.0.0",
        "accelerate==1.0.0",
        "wandb",
    ],
    pip_extra_index_url=["https://download.pytorch.org/whl/cu121"],
    apt_packages=["git"],
    env={"WANDB_PROJECT": "my-project"},
    registry="ghcr.io/my-org",
    source_copy_mode=CopyFileDetection.ALL,  # copy local source into image
)

# Lightweight CPU image for data preprocessing
preprocess_image = ImageSpec(
    name="preprocess",
    packages=["pandas", "pyarrow", "datasets"],
    registry="ghcr.io/my-org",
)

@task(container_image=training_image)
def train_model(config: dict) -> str:
    ...

@task(container_image=preprocess_image)
def preprocess_data(path: str) -> str:
    ...
```

### ImageSpec with is_container()

Avoid importing heavy dependencies at module level:

```python
training_image = ImageSpec(packages=["torch", "transformers"], ...)

if training_image.is_container():
    import torch
    from transformers import AutoModelForCausalLM

@task(container_image=training_image)
def train(model_name: str) -> float:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ...
```

`is_container()` returns `True` only when running inside the built image — prevents import errors during compilation on your laptop.

### Builder Backends

| Builder | Description | When to Use |
|---|---|---|
| `default` | Docker build | Standard, requires Docker daemon |
| `envd` | envd builder | Faster caching, parallel builds |
| `noop` | Skip building | Pre-built images, CI/CD handles builds |

## Tasks

### @task Decorator Settings

| Setting | Purpose | Default |
|---|---|---|
| `container_image` | ImageSpec or image string | Default flytekit image |
| `requests` | Resource requests | None |
| `limits` | Resource limits | None |
| `retries` | Max retry count | `0` |
| `timeout` | Task timeout | None |
| `cache` | Enable output caching | `False` |
| `cache_version` | Cache version string | `""` |
| `cache_serialize` | Serialize cache access | `False` |
| `interruptible` | Can run on preemptible nodes | None |
| `environment` | Extra env vars | `{}` |
| `secret_requests` | Secrets to mount | `[]` |
| `pod_template` | Custom PodTemplate | None |
| `accelerator` | GPU accelerator type | None |
| `task_config` | Plugin config (Spark, Ray, etc.) | None |

### Resource Configuration

```python
from flytekit import task, Resources

@task(
    requests=Resources(cpu="4", mem="16Gi", gpu="1", ephemeral_storage="50Gi"),
    limits=Resources(cpu="8", mem="32Gi", gpu="1"),
    accelerator=GPUAccelerator("nvidia-tesla-a100"),
    timeout=timedelta(hours=12),
    retries=2,
    interruptible=True,  # allow scheduling on spot/preemptible nodes
)
def train_model(data_path: str) -> float:
    ...
```

### Retries and Interruptibility

```python
@task(
    retries=3,                    # total attempts = retries + 1 on user errors
    interruptible=True,           # retries on preemption don't count against retries budget
)
def training_task(config: dict) -> float:
    ...
```

- `retries` handles user-code failures (exceptions)
- `interruptible=True` retries on node preemption separately (configured cluster-wide)
- System errors (OOM killed, node failure) have separate retry budgets

### PodTemplate for Advanced Pod Config

```python
from flytekit import task, PodTemplate
from kubernetes.client import V1PodSpec, V1Container, V1VolumeMount, V1Volume, V1EmptyDirVolumeSource

custom_pod = PodTemplate(
    pod_spec=V1PodSpec(
        containers=[
            V1Container(
                name="primary",
                volume_mounts=[
                    V1VolumeMount(name="shm", mount_path="/dev/shm"),
                    V1VolumeMount(name="data", mount_path="/data"),
                ],
            ),
        ],
        volumes=[
            V1Volume(name="shm", empty_dir=V1EmptyDirVolumeSource(medium="Memory", size_limit="16Gi")),
            V1Volume(name="data", persistent_volume_claim={"claimName": "training-data"}),
        ],
        node_selector={"nvidia.com/gpu.product": "NVIDIA-A100-SXM4-80GB"},
    ),
)

@task(pod_template=custom_pod, requests=Resources(gpu="1"))
def gpu_training(config: dict) -> str:
    ...
```

## Type System

Flyte's type system enables data passing between tasks with automatic serialization.

| Type | Purpose | Backed By |
|---|---|---|
| `int`, `float`, `str`, `bool` | Primitives | Protobuf literals |
| `list[T]`, `dict[str, T]` | Collections | Protobuf |
| `FlyteFile` | Single file | Blob store (S3/GCS) |
| `FlyteDirectory` | Directory of files | Blob store prefix |
| `StructuredDataset` | Typed tabular data | Parquet in blob store |
| `@dataclass` | Structured records | Protobuf Struct |
| `Annotated[T, ...]` | Type with metadata | Depends on T |
| `Enum` | Enumerated values | Protobuf |

### StructuredDataset (for large data)

```python
from flytekit.types.structured import StructuredDataset
import pandas as pd

@task
def generate_data() -> StructuredDataset:
    df = pd.DataFrame({"col": [1, 2, 3]})
    return StructuredDataset(dataframe=df)

@task
def consume_data(ds: StructuredDataset) -> float:
    df = ds.open(pd.DataFrame).all()  # reads from blob store
    return df["col"].mean()
```

### FlyteFile and FlyteDirectory

```python
from flytekit.types.file import FlyteFile
from flytekit.types.directory import FlyteDirectory

@task
def train(data_dir: FlyteDirectory, config: FlyteFile) -> FlyteFile:
    # data_dir and config are automatically downloaded to local paths
    local_data = data_dir.download()
    model_path = "/tmp/model.pt"
    torch.save(model.state_dict(), model_path)
    return FlyteFile(model_path)  # automatically uploaded to blob store
```

## Workflows

### @workflow — Static DAG

```python
from flytekit import workflow

@workflow
def training_pipeline(model_name: str, data_path: str) -> float:
    processed = preprocess_data(path=data_path)
    model = train_model(data_path=processed)
    metrics = evaluate_model(model=model)
    return metrics
```

**Constraints:** No Python if/else, loops, or side effects. Use `conditional` for branching:

```python
from flytekit import conditional

@workflow
def pipeline(data_path: str, use_large_model: bool) -> float:
    return (
        conditional("model_size")
        .if_(use_large_model.is_true())
        .then(train_large(data_path=data_path))
        .else_()
        .then(train_small(data_path=data_path))
    )
```

### @dynamic — Runtime DAG Construction

Dynamic workflows execute Python at runtime to build a DAG. Runs in a pod:

```python
from flytekit import dynamic

@dynamic
def hyperparameter_search(configs: list[dict]) -> list[float]:
    results = []
    for config in configs:  # Python loops work here
        result = train_model(config=config)
        results.append(result)
    return results
```

### @eager — Full Python Control (Experimental)

Eager workflows run Python with full control flow, executing tasks as they're called:

```python
from flytekit.experimental import eager

@eager
async def adaptive_training(model_name: str) -> float:
    metrics = await initial_train(model_name=model_name)
    if metrics < 0.9:  # real Python if/else
        metrics = await extended_train(model_name=model_name)
    return metrics
```

### map_task — Parallel Fan-Out

```python
from flytekit import map_task

@workflow
def parallel_eval(models: list[str]) -> list[float]:
    return map_task(evaluate_model)(model_path=models)
```

`map_task` creates a K8s array task — each element runs as a separate pod.

## LaunchPlans

LaunchPlans are named, versioned configurations for executing workflows:

```python
from flytekit import LaunchPlan, CronSchedule, FixedInputs

# Scheduled execution
nightly_train = LaunchPlan.get_or_create(
    training_pipeline,
    name="nightly-training",
    schedule=CronSchedule(schedule="0 2 * * *"),  # 2 AM daily
    fixed_inputs={"model_name": "llama-8b"},
    default_inputs={"data_path": "/data/latest"},
    max_parallelism=10,
    labels={"team": "ml"},
)

# Fixed config for production
prod_launch = LaunchPlan.get_or_create(
    training_pipeline,
    name="prod-training",
    fixed_inputs={"model_name": "llama-70b", "data_path": "/data/prod"},
)
```

### LaunchPlan Settings

| Setting | Purpose |
|---|---|
| `schedule` | CronSchedule or FixedRate |
| `fixed_inputs` | Inputs locked at registration |
| `default_inputs` | Defaults that can be overridden |
| `max_parallelism` | Max concurrent task executions |
| `labels` | K8s labels for the execution |
| `annotations` | K8s annotations |
| `notifications` | Email/Slack on completion/failure |
| `raw_output_data_config` | Override output blob store location |

## Secrets

```python
from flytekit import task, Secret

@task(
    secret_requests=[
        Secret(group="wandb", key="api_key", mount_requirement=Secret.MountType.ENV_VAR),
        Secret(group="hf", key="token"),
    ],
)
def train_with_secrets(config: dict) -> str:
    import os
    wandb_key = os.environ["WANDB_API_KEY"]  # ENV_VAR mount
    # or: flytekit.current_context().secrets.get("wandb", "api_key")
    ...
```

Secrets are backed by K8s Secrets in the Flyte namespace. Create them with:
```
kubectl create secret generic wandb --from-literal=api_key=<key> -n <flyte-ns>
```

## Intra-Task Checkpointing

For long-running tasks, checkpoint progress for fault tolerance:

```python
from flytekit import task, current_context

@task(retries=3)
def long_training(epochs: int) -> float:
    ctx = current_context()
    checkpoint = ctx.checkpoint

    # Try to restore
    prev = checkpoint.read()
    start_epoch = 0
    if prev:
        state = pickle.loads(prev)
        model.load_state_dict(state["model"])
        start_epoch = state["epoch"] + 1

    for epoch in range(start_epoch, epochs):
        train_one_epoch(model)
        checkpoint.write(pickle.dumps({"model": model.state_dict(), "epoch": epoch}))

    return evaluate(model)
```

## Local Testing

```python
# Tasks are regular Python functions locally
result = train_model(config={"lr": 1e-4})

# Workflows too
metrics = training_pipeline(model_name="test", data_path="/local/data")
```

No cluster needed. Types are resolved locally (FlyteFile → local path, etc.).

## Registration and Packaging

```bash
# Register all workflows/tasks in current directory to Flyte cluster
# Run as a CI/CD step or Job — not interactively
# pyflyte register --project my-project --domain development .

# Package to a tarball (for offline registration)
# pyflyte package --image ghcr.io/my-org/training:latest -o workflows.tgz
```

## Debugging

See `references/troubleshooting.md` for:
- Serialization errors during registration
- Pod failures and resource issues
- Type mismatch errors
- ImageSpec build failures
- Checkpoint and caching issues

## Cross-References

- [flyte-deployment](../flyte-deployment/) — Deploy and operate Flyte on Kubernetes
- [flyte-kuberay](../flyte-kuberay/) — Run Ray workloads as Flyte tasks
- [huggingface-transformers](../huggingface-transformers/) — HF models/datasets in Flyte tasks
- [wandb](../wandb/) — Log experiment metrics from Flyte tasks to W&B

## Reference

- [Flytekit API docs](https://docs.flyte.org/en/latest/api/flytekit/)
- [Flytekit GitHub](https://github.com/flyteorg/flytekit)
- [Flyte user guide](https://docs.flyte.org/en/latest/user_guide/)
- [FlytePropeller architecture](https://docs-legacy.flyte.org/en/latest/user_guide/concepts/component_architecture/flytepropeller_architecture.html)
- `references/troubleshooting.md` — common errors and fixes
