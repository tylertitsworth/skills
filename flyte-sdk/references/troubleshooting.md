# Flytekit Troubleshooting

## Serialization Errors

### "Failed to convert literal"

**Cause**: Task return type doesn't match annotation.

```
TypeError: Failed to convert literal to type <class 'list'>
```

**Fix**: Ensure the actual return value matches the declared type exactly:
```python
# Wrong — returns dict but declares list
@task
def bad() -> list[str]:
    return {"a": "b"}  # TypeError

# Right
@task
def good() -> list[str]:
    return ["a", "b"]
```

### "Could not serialize unknown type"

**Cause**: Using a type Flyte doesn't know how to serialize (e.g., custom class without `@dataclass`).

**Fix**: Use `@dataclass` for custom types, or register a custom TypeTransformer:
```python
# Wrong
class MyConfig:
    lr: float = 0.01

# Right
from dataclasses import dataclass

@dataclass
class MyConfig:
    lr: float = 0.01
```

**Supported types**: `int`, `float`, `str`, `bool`, `bytes`, `datetime`, `timedelta`, `list`, `dict`, `typing.Optional`, `enum.Enum`, `@dataclass`, `FlyteFile`, `FlyteDirectory`, `StructuredDataset`, `pandas.DataFrame`, `pyarrow.Table`.

### "Cannot pickle" errors

**Cause**: Task inputs/outputs contain unpicklable objects (open file handles, locks, generators).

**Fix**: Return serializable data. For large objects, write to disk and return `FlyteFile` or `FlyteDirectory`:
```python
# Wrong
@task
def bad() -> object:
    return open("/tmp/data.csv")

# Right
@task
def good() -> FlyteFile:
    return FlyteFile(path="/tmp/data.csv")
```

## Type Mismatch Errors

### "Expected type X but got type Y"

Flyte checks types at compile time (registration) and runtime. Common causes:

```python
# Mismatched nested types
@task
def producer() -> list[int]:
    return [1, 2, 3]

@task
def consumer(data: list[float]) -> float:  # list[float] != list[int]
    return sum(data)

@workflow
def wf() -> float:
    data = producer()
    return consumer(data=data)  # Type error at registration!
```

**Fix**: Keep types consistent across task boundaries. Use `list[float]` in both or convert explicitly.

### Optional type gotchas

```python
# Wrong — None isn't Optional unless declared
@task
def maybe_result(flag: bool) -> str:
    if flag:
        return "done"
    return None  # Runtime error!

# Right
@task
def maybe_result(flag: bool) -> Optional[str]:
    if flag:
        return "done"
    return None
```

## Import Errors with ImageSpec

### "ModuleNotFoundError" at runtime

**Cause**: Package installed in ImageSpec but import is at module top-level, which fails during serialization (which runs in your local env).

**Fix**: Guard imports behind `is_container()`:
```python
gpu_image = ImageSpec(
    packages=["transformers", "torch"],
    base_image="nvcr.io/nvidia/pytorch:24.01-py3",
    registry="ghcr.io/myorg",
)

# Wrong — fails during pyflyte serialize if transformers not installed locally
from transformers import AutoModel

# Right — only imports inside the container
if gpu_image.is_container():
    from transformers import AutoModel

@task(container_image=gpu_image)
def train():
    model = AutoModel.from_pretrained("bert-base-uncased")
    ...
```

### ImageSpec build failures

**Common causes:**
- Missing `registry` — ImageSpec needs a push target for remote execution
- Network issues during pip install — use `pip_extra_index_url` for private repos
- Conflicting package versions — pin versions explicitly

```python
# Debug: build the image locally
# pyflyte will build automatically, but you can test:
ImageSpec(
    packages=["torch==2.4.0", "transformers==4.44.0"],
    base_image="python:3.11-slim",
    registry="ghcr.io/myorg",
    python_version="3.11",
).force_build()
```

## Promise Errors in Workflows

### "Promise has no attribute X"

**Cause**: Trying to access attributes of a promise (task output) inside a workflow body.

```python
@workflow
def bad_wf():
    result = my_task()
    if result > 0.5:  # ERROR — result is a Promise, not a float
        ...
```

**Fix**: Use `conditional()` or move logic into a task:
```python
# Option 1: conditional
@workflow
def good_wf():
    result = my_task()
    return conditional("check").if_(result > 0.5, then=...).else_(...)

# Option 2: move logic to a task
@task
def decide(result: float) -> str:
    if result > 0.5:
        return "deploy"
    return "retrain"

@workflow
def good_wf():
    result = my_task()
    decision = decide(result=result)
    ...
```

### "Cannot use len() on Promise"

Same issue — use a task to compute length:
```python
@task
def get_length(items: list[str]) -> int:
    return len(items)
```

## Registration Failures

### "pyflyte register" hangs or fails

1. **Check Flyte endpoint**: `pyflyte --config ~/.flyte/config.yaml status`
2. **Auth issues**: Ensure valid credentials in `~/.flyte/config.yaml`
3. **Version conflict**: Use `--version` to avoid collisions:
   ```bash
   pyflyte register --project myproject --domain dev --version $(git rev-parse --short HEAD) ./workflows/
   ```

### "Entity already registered"

Flyte is immutable by default — same version + entity = error.

**Fix**: Bump the version or use a unique identifier:
```bash
pyflyte register --version "v2-$(date +%s)" ./workflows/
```

## Execution Failures

### Task OOMKilled

Pod ran out of memory. Increase `requests` and `limits`:
```python
@task(
    requests=Resources(mem="32Gi"),
    limits=Resources(mem="64Gi"),
)
```

For GPU OOM, reduce batch size or enable gradient checkpointing — this is a model-level issue, not Flyte-specific.

### Task stuck in "Queued"

Check:
1. **Resource quotas**: Does the namespace have enough GPU/CPU quota?
2. **Node availability**: Are nodes with the requested accelerator available?
3. **Tolerations**: Does the pod template include tolerations matching node taints?
4. **Kueue integration**: If using Kueue, check ClusterQueue admission.

### Spot/preemptible task keeps restarting

`interruptible=True` tasks get rescheduled on preemption. Combine with checkpointing:
```python
@task(interruptible=True, retries=5)
def resilient_train(epochs: int) -> float:
    ctx = current_context()
    checkpoint = ctx.checkpoint
    prev = checkpoint.read()
    start = deserialize(prev)["epoch"] if prev else 0
    for epoch in range(start, epochs):
        train_epoch(epoch)
        checkpoint.write(serialize({"epoch": epoch + 1}))
```

## pyflyte CLI Issues

### "No module named flytekit"

Install flytekit: `pip install flytekit` or `uv add flytekit`

### "pyflyte run" works but "pyflyte register" fails

`run` executes locally; `register` serializes and uploads. Serialization is stricter:
- All imports must be resolvable (or guarded with `is_container()`)
- All types must be Flyte-serializable
- Circular imports cause failures

### Config file location

```yaml
# ~/.flyte/config.yaml
admin:
  endpoint: dns:///flyte.example.com
  insecure: false
  authType: Pkce  # or ClientSecret
```

Override: `FLYTE_CONFIG=path/to/config.yaml pyflyte ...`
