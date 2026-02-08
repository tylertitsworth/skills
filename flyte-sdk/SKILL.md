---
name: flyte-sdk
description: >
  Write Flyte workflows with Flytekit — the Python SDK for tasks, workflows, launch plans,
  and data types. Use when: (1) Writing @task and @workflow functions with type annotations,
  (2) Using Flyte types (FlyteFile, FlyteDirectory, StructuredDataset), (3) Building dynamic
  workflows or map tasks for parallel execution, (4) Configuring per-task resources (GPU, memory,
  tolerations), (5) Creating launch plans with schedules and notifications, (6) Setting up
  ImageSpec for reproducible container builds, (7) Testing workflows locally before cluster
  registration, (8) Debugging serialization or type mismatch errors.
---

# Flytekit SDK

Flytekit (`flytekit`) is the Python SDK for authoring Flyte tasks and workflows. Version: **1.16.x**.

> **Companion skill**: See `flyte-deployment` for installing and operating Flyte on Kubernetes.
> This skill covers **writing workflows** with the SDK.

## Core Concepts

### Tasks

Tasks are containerized units of compute — regular Python functions decorated with `@task`:

```python
from flytekit import task

@task(cache=True, cache_version="1", retries=3)
def preprocess(data_path: str, batch_size: int = 32) -> list[dict]:
    """Tasks must have type-annotated inputs and outputs."""
    # runs in its own K8s pod on a Flyte cluster
    ...
    return processed_records
```

Key `@task` parameters:

| Parameter | Purpose | Example |
|-----------|---------|---------|
| `cache` / `cache_version` | Memoize outputs by inputs | `cache=True, cache_version="2"` |
| `retries` | Auto-retry on failure | `retries=3` |
| `timeout` | Max execution time | `timeout=timedelta(hours=2)` |
| `requests` / `limits` | CPU/memory/GPU resources | See [Resources](#resources) |
| `container_image` | Custom image | `container_image=gpu_image` |
| `accelerator` | GPU accelerator type | `accelerator=A100` |
| `secret_requests` | Mount secrets | `secret_requests=[Secret(group="aws")]` |
| `interruptible` | Allow spot/preemptible nodes | `interruptible=True` |

### Workflows

Workflows compose tasks into a DAG. They're compiled at registration time — the body is a **DSL**, not regular Python:

```python
from flytekit import workflow

@workflow
def training_pipeline(data_path: str, epochs: int = 10) -> float:
    prepared = preprocess(data_path=data_path)
    model = train_model(data=prepared, epochs=epochs)
    return evaluate(model=model)
```

**Critical rules for workflow bodies:**
- Task outputs are **promises** (placeholders), not actual values
- You **cannot** use `if/else` on promises — use `flytekit.conditional` instead
- No `print()`, `len()`, or attribute access on promises
- Only pass promises to other tasks/workflows
- Use `>>` operator for ordering without data dependencies: `task_a() >> task_b()`

### Workflow Conditionals

```python
from flytekit import conditional

@workflow
def conditional_wf(x: float) -> float:
    return (
        conditional("check_threshold")
        .if_(x > 0.9, then=deploy_model(score=x))
        .else_(retrain_model(score=x))
    )
```

## Type System

Flyte has its own type system that maps to/from Python types. All task I/O must use supported types.

### Primitive Types

`int`, `float`, `str`, `bool`, `datetime.datetime`, `datetime.timedelta`, `bytes`

### Collection Types

`list[T]`, `dict[str, T]`, `typing.Optional[T]`

### Structured Types

```python
from dataclasses import dataclass
from flytekit.types.file import FlyteFile
from flytekit.types.directory import FlyteDirectory
from flytekit.types.structured import StructuredDataset

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    batch_size: int = 32
    model_name: str = "bert-base-uncased"

@task
def train(config: TrainingConfig, dataset: FlyteFile) -> FlyteDirectory:
    """
    FlyteFile: references a single file (local or remote, e.g. s3://).
    FlyteDirectory: references a directory of files.
    Both auto-handle upload/download between tasks.
    """
    local_path = dataset.download()
    output_dir = FlyteDirectory.new_remote()
    # ... training logic ...
    return FlyteDirectory(path="/tmp/model_output")

@task
def load_data(uri: str) -> StructuredDataset:
    """StructuredDataset: typed columnar data (parquet, Arrow, etc.)."""
    import pandas as pd
    df = pd.read_parquet(uri)
    return StructuredDataset(dataframe=df)
```

**FlyteFile with format hints:**
```python
from flytekit.types.file import FlyteFile

# Typed file references
csv_file: FlyteFile[typing.TypeVar("csv")]
onnx_model: FlyteFile[typing.TypeVar("onnx")]
```

### Enum Types

```python
import enum

class ModelArch(enum.Enum):
    BERT = "bert"
    GPT = "gpt"
    LLAMA = "llama"

@task
def select_model(arch: ModelArch) -> str:
    return arch.value
```

## Resources

Configure per-task compute resources:

```python
from flytekit import Resources
from flytekit.extras.accelerators import A100, T4, GPUAccelerator

@task(
    requests=Resources(cpu="4", mem="16Gi", gpu="1", ephemeral_storage="50Gi"),
    limits=Resources(cpu="8", mem="32Gi", gpu="1", ephemeral_storage="100Gi"),
    accelerator=A100,  # or T4, L4, GPUAccelerator("nvidia-a100-80gb")
)
def train_on_gpu(data: FlyteFile) -> FlyteDirectory:
    import torch
    assert torch.cuda.is_available()
    ...
```

**Pod-level configuration with tolerations and node selectors:**

```python
from flytekit import task, PodTemplate
from kubernetes.client import V1PodSpec, V1Container, V1Toleration

gpu_pod = PodTemplate(
    pod_spec=V1PodSpec(
        tolerations=[
            V1Toleration(
                key="nvidia.com/gpu",
                operator="Exists",
                effect="NoSchedule",
            )
        ],
        node_selector={"gpu-type": "a100"},
        containers=[V1Container(name="primary")],
    )
)

@task(pod_template=gpu_pod, requests=Resources(gpu="1"))
def gpu_task() -> None:
    ...
```

## ImageSpec — Reproducible Environments

`ImageSpec` builds container images without writing Dockerfiles:

```python
from flytekit import ImageSpec, task

gpu_image = ImageSpec(
    name="ml-training",
    base_image="nvcr.io/nvidia/pytorch:24.01-py3",
    packages=[
        "transformers==4.44.0",
        "datasets==3.0.0",
        "accelerate==0.34.0",
    ],
    conda_packages=["cudatoolkit=12.1"],
    registry="ghcr.io/myorg",
    python_version="3.11",
)

# Guard imports that only exist inside the custom image
if gpu_image.is_container():
    from transformers import AutoModelForCausalLM

@task(container_image=gpu_image, requests=Resources(gpu="1"))
def fine_tune(model_name: str) -> FlyteDirectory:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ...
```

**Key `ImageSpec` fields:**

| Field | Purpose |
|-------|---------|
| `base_image` | Starting image (NGC, Docker Hub, etc.) |
| `packages` | pip packages to install |
| `conda_packages` | conda packages |
| `apt_packages` | System packages via apt |
| `registry` | Container registry to push to |
| `python_version` | Python version override |
| `source_root` | Copy local source code into image |
| `commands` | Arbitrary RUN commands |
| `pip_extra_index_url` | Additional PyPI indexes |

## Dynamic Workflows and Map Tasks

### Dynamic Workflows

`@dynamic` creates the DAG at **runtime** — use when the structure depends on data:

```python
from flytekit import dynamic

@dynamic(container_image=gpu_image)
def hyperparameter_search(
    configs: list[TrainingConfig],
) -> list[float]:
    results = []
    for cfg in configs:
        score = train_and_eval(config=cfg)
        results.append(score)
    return results
```

### Map Tasks

`map_task` runs the same task over a list of inputs **in parallel**:

```python
from flytekit import map_task

@task
def train_single(config: TrainingConfig) -> float:
    ...

@workflow
def parallel_training(configs: list[TrainingConfig]) -> list[float]:
    return map_task(train_single)(config=configs)
```

**Map task with concurrency limit:**
```python
from flytekit import map_task

# Max 4 parallel pods (e.g., limited GPU quota)
map_task(train_single, concurrency=4)(config=configs)
```

## Launch Plans

Launch plans bind default inputs and optionally add schedules:

```python
from flytekit import LaunchPlan, CronSchedule, FixedRate
from datetime import timedelta

# Default inputs
nightly_plan = LaunchPlan.get_or_create(
    workflow=training_pipeline,
    name="nightly_retrain",
    default_inputs={"data_path": "s3://data/latest", "epochs": 5},
)

# Cron schedule
scheduled_plan = LaunchPlan.get_or_create(
    workflow=training_pipeline,
    name="scheduled_retrain",
    default_inputs={"data_path": "s3://data/latest", "epochs": 5},
    schedule=CronSchedule(schedule="0 2 * * *", kickoff_time_input_arg=None),
)

# Fixed-rate schedule
hourly_plan = LaunchPlan.get_or_create(
    workflow=training_pipeline,
    name="hourly_eval",
    default_inputs={"data_path": "s3://data/latest", "epochs": 1},
    schedule=FixedRate(duration=timedelta(hours=1)),
)
```

**Launch plan notifications:**
```python
from flytekit import Email, Slack
from flytekit.models.core.execution import WorkflowExecutionPhase

notified_plan = LaunchPlan.get_or_create(
    workflow=training_pipeline,
    name="notified_retrain",
    default_inputs={"data_path": "s3://data/latest"},
    notifications=[
        Email(
            phases=[WorkflowExecutionPhase.FAILED],
            recipients_email=["team@example.com"],
        ),
        Slack(
            phases=[WorkflowExecutionPhase.SUCCEEDED],
            recipients_email=["#ml-alerts"],  # Slack channel
        ),
    ],
)
```

## Intra-Task Checkpointing

For long-running tasks that should resume after failures:

```python
from flytekit import task, current_context

@task(retries=3)
def long_training(epochs: int) -> float:
    ctx = current_context()
    checkpoint = ctx.checkpoint

    # Try to restore from previous attempt
    prev = checkpoint.read()
    start_epoch = 0
    if prev:
        state = deserialize(prev)
        start_epoch = state["epoch"]
        model.load_state_dict(state["weights"])

    for epoch in range(start_epoch, epochs):
        loss = train_one_epoch(model, epoch)
        # Checkpoint periodically
        checkpoint.write(serialize({"epoch": epoch + 1, "weights": model.state_dict()}))

    return loss
```

## Local Testing

Always test locally before registering:

```bash
# Run a single task
pyflyte run my_workflows.py preprocess --data_path /tmp/data --batch_size 16

# Run a full workflow
pyflyte run my_workflows.py training_pipeline --data_path /tmp/data --epochs 2

# Run from a remote file
pyflyte run --remote my_workflows.py training_pipeline --data_path s3://bucket/data

# Register all workflows in a package
pyflyte register --project my_project --domain development ./workflows/

# Serialize without registering (for CI)
pyflyte serialize workflows/ -o /tmp/output
```

**Local execution bypasses the Flyte cluster** — tasks run as regular Python functions in your process. This means:
- `Resources`, `accelerator`, and `pod_template` are ignored locally
- `FlyteFile`/`FlyteDirectory` work with local paths
- `cache` still works (uses local SQLite)

## Secrets

```python
from flytekit import task, Secret

@task(
    secret_requests=[
        Secret(group="aws", key="access_key"),
        Secret(group="aws", key="secret_key"),
    ]
)
def download_from_s3(bucket: str) -> FlyteFile:
    ctx = current_context()
    access_key = ctx.secrets.get("aws", "access_key")
    secret_key = ctx.secrets.get("aws", "secret_key")
    ...
```

## Common Patterns

### ML Training Pipeline (End-to-End)

```python
from flytekit import task, workflow, Resources, ImageSpec
from flytekit.types.file import FlyteFile
from flytekit.types.directory import FlyteDirectory
from flytekit.extras.accelerators import A100
from dataclasses import dataclass

gpu_image = ImageSpec(
    base_image="nvcr.io/nvidia/pytorch:24.01-py3",
    packages=["transformers", "datasets", "wandb"],
    registry="ghcr.io/myorg",
)

@dataclass
class TrainConfig:
    model_name: str = "meta-llama/Llama-3.1-8B"
    learning_rate: float = 2e-5
    epochs: int = 3
    batch_size: int = 4

@task(container_image=gpu_image, requests=Resources(cpu="4", mem="16Gi"))
def prepare_data(dataset_name: str) -> FlyteDirectory:
    from datasets import load_dataset
    ds = load_dataset(dataset_name)
    ds.save_to_disk("/tmp/prepared")
    return FlyteDirectory(path="/tmp/prepared")

@task(
    container_image=gpu_image,
    requests=Resources(cpu="8", mem="64Gi", gpu="1"),
    accelerator=A100,
    timeout=timedelta(hours=6),
    interruptible=True,
)
def fine_tune(config: TrainConfig, data_dir: FlyteDirectory) -> FlyteDirectory:
    from transformers import Trainer, TrainingArguments
    local_data = data_dir.download()
    # ... set up trainer ...
    trainer.train()
    trainer.save_model("/tmp/model")
    return FlyteDirectory(path="/tmp/model")

@task(container_image=gpu_image, requests=Resources(cpu="4", mem="16Gi", gpu="1"))
def evaluate(model_dir: FlyteDirectory) -> float:
    # ... load model, run eval ...
    return accuracy

@workflow
def llm_finetune_pipeline(
    dataset_name: str = "tatsu-lab/alpaca",
    config: TrainConfig = TrainConfig(),
) -> float:
    data = prepare_data(dataset_name=dataset_name)
    model = fine_tune(config=config, data_dir=data)
    return evaluate(model_dir=model)
```

### Branching with Conditionals

```python
from flytekit import conditional

@workflow
def deploy_or_retrain(dataset: str) -> str:
    score = llm_finetune_pipeline(dataset_name=dataset)
    return (
        conditional("quality_gate")
        .if_(score >= 0.85, then=deploy_model(score=score))
        .else_(notify_team(score=score))
    )
```

## Debugging

See `references/troubleshooting.md` for:
- Serialization errors and type mismatches
- Import errors with ImageSpec
- Promise-related mistakes in workflows
- Registration and execution failures
- Common `pyflyte` CLI issues

## Reference

- [Flytekit API docs](https://docs.flyte.org/projects/flytekit/en/latest/)
- [Flytekit GitHub](https://github.com/flyteorg/flytekit)
- [Flyte user guide (legacy)](https://docs-legacy.flyte.org/en/latest/user_guide/index.html)
- `references/troubleshooting.md` — common errors and fixes
