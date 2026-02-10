---
name: ray-train
description: >
  Distributed model training across GPUs and nodes with Ray Train. Use when:
  (1) Configuring distributed training with TorchTrainer (scaling, resources, placement),
  (2) Setting up fault tolerance (checkpointing, failure recovery, elastic training),
  (3) Integrating with HuggingFace Transformers Trainer,
  (4) Configuring data ingestion with Ray Data for streaming pipelines,
  (5) Tuning hyperparameters with Ray Tune integration,
  (6) Configuring TorchConfig (backend, timeout, communication),
  (7) Multi-node training on Kubernetes (RayJob/RayCluster),
  (8) Debugging training issues (communication errors, OOM, slow data loading, rank mismatches).
---

# Ray Train

Ray Train provides distributed training built on Ray, abstracting away distributed setup and adding fault tolerance, checkpointing, and integration with Ray Data and Ray Tune. Version: **2.53.0+**.

## TorchTrainer Configuration

### Core Settings

```python
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig, FailureConfig

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=ScalingConfig(...),
    run_config=RunConfig(...),
    torch_config=TorchConfig(...),
    datasets={"train": ray_dataset},        # optional Ray Data integration
    dataset_config=DataConfig(...),          # optional data config
    resume_from_checkpoint=checkpoint,       # resume from saved checkpoint
)
result = trainer.fit()
```

### ScalingConfig

| Setting | Purpose | Default |
|---|---|---|
| `num_workers` | Total training workers (each gets a GPU) | required |
| `use_gpu` | Assign GPUs to workers | `False` |
| `resources_per_worker` | Resource dict per worker | `{}` |
| `placement_strategy` | `SPREAD`, `PACK`, `STRICT_SPREAD`, `STRICT_PACK` | `PACK` |
| `trainer_resources` | Resources for the trainer coordinator | `{"CPU": 0}` |
| `accelerator_type` | Required accelerator (e.g., `A100`, `H100`) | None |

```python
scaling_config = ScalingConfig(
    num_workers=8,
    use_gpu=True,
    resources_per_worker={"CPU": 8, "GPU": 1},
    placement_strategy="PACK",           # co-locate workers for faster communication
    accelerator_type="A100",             # require specific GPU type
)
```

**Placement strategies:**
- `PACK` — co-locate workers on same node (best for TP, faster communication)
- `SPREAD` — distribute across nodes (better fault isolation)
- `STRICT_PACK` / `STRICT_SPREAD` — hard constraints (fail if can't satisfy)

## Fault Tolerance and Resiliency

### CheckpointConfig

```python
checkpoint_config = CheckpointConfig(
    num_to_keep=3,                       # keep only last N checkpoints
    checkpoint_score_attribute="eval_loss",  # metric to rank checkpoints
    checkpoint_score_order="min",        # "min" or "max"
)
```

### FailureConfig

```python
from ray.train import FailureConfig

failure_config = FailureConfig(
    max_failures=3,                      # auto-restart up to 3 times on failure
    fail_fast=False,                     # True = fail immediately on any worker error
)
```

When a worker fails, Ray Train:
1. Stops all workers
2. Restores from the latest checkpoint
3. Restarts all workers from that checkpoint
4. Continues training

### Checkpointing in the Training Loop

```python
from ray.train import Checkpoint
import tempfile, os

def train_func(config):
    model = ...
    optimizer = ...
    
    # Resume from checkpoint if available
    checkpoint = ray.train.get_checkpoint()
    start_epoch = 0
    if checkpoint:
        with checkpoint.as_directory() as ckpt_dir:
            state = torch.load(os.path.join(ckpt_dir, "model.pt"))
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = state["epoch"] + 1
    
    for epoch in range(start_epoch, num_epochs):
        train_one_epoch(model, optimizer)
        
        # Save checkpoint (only rank 0 saves, but all ranks must call report)
        with tempfile.TemporaryDirectory() as tmp:
            if ray.train.get_context().get_world_rank() == 0:
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }, os.path.join(tmp, "model.pt"))
            
            ray.train.report(
                metrics={"loss": loss, "epoch": epoch},
                checkpoint=Checkpoint.from_directory(tmp),
            )
```

### Storage Configuration

```python
from ray.train import RunConfig

run_config = RunConfig(
    name="my-training-run",
    storage_path="s3://my-bucket/ray-results",   # or /mnt/shared-pvc
    checkpoint_config=checkpoint_config,
    failure_config=failure_config,
    log_to_file=True,
)
```

**Storage requirements:**
- All workers must be able to read/write to `storage_path`
- On K8s: use a shared PVC or S3/GCS with credentials in the container env
- Checkpoint upload is async by default

## RunConfig

| Setting | Purpose | Default |
|---|---|---|
| `name` | Run name (directory name) | Auto-generated |
| `storage_path` | Where to save checkpoints/results | `~/ray_results` |
| `checkpoint_config` | Checkpoint settings | `CheckpointConfig()` |
| `failure_config` | Failure handling | `FailureConfig()` |
| `log_to_file` | Redirect stdout/stderr to files | `False` |
| `stop` | Stopping criteria | None |
| `callbacks` | List of callbacks | `[]` |
| `sync_config` | Sync settings for cloud storage | Auto |

## TorchConfig

| Setting | Purpose | Default |
|---|---|---|
| `backend` | Distributed backend | `"nccl"` (GPU), `"gloo"` (CPU) |
| `timeout_s` | Timeout for collective operations (seconds) | `1800` |

```python
from ray.train.torch import TorchConfig

torch_config = TorchConfig(
    backend="nccl",
    timeout_s=3600,          # increase for large models or slow checkpointing
)
```

## HuggingFace Transformers Integration

```python
from ray.train.huggingface.transformers import prepare_trainer, RayTrainReportCallback

def train_func(config):
    from transformers import Trainer, TrainingArguments
    
    training_args = TrainingArguments(
        output_dir="/tmp/hf-output",
        per_device_train_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        learning_rate=config["lr"],
        bf16=True,
        gradient_accumulation_steps=config.get("grad_accum", 1),
        save_strategy="epoch",
        logging_steps=10,
        # Do NOT set deepspeed or fsdp here — Ray Train handles distribution
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=[RayTrainReportCallback()],  # reports metrics + checkpoints to Ray
    )
    
    trainer = prepare_trainer(trainer)  # sets up distributed training
    trainer.train()
```

**Key:** Use `prepare_trainer()` instead of configuring DeepSpeed/FSDP directly in TrainingArguments. Ray Train handles the distributed setup.

## Data Ingestion with Ray Data

### Streaming Data Pipeline

```python
import ray

# Create a streaming dataset (doesn't load all data into memory)
ds = ray.data.read_parquet("s3://my-bucket/training-data/")

# Preprocessing pipeline
ds = ds.map(tokenize_function)
ds = ds.random_shuffle()

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=ScalingConfig(num_workers=8, use_gpu=True),
    datasets={"train": ds, "eval": eval_ds},
)
```

### DataConfig

```python
from ray.train import DataConfig

data_config = DataConfig(
    datasets_to_split="all",             # split across workers (default for train)
)
```

### In the Training Loop

```python
def train_func(config):
    train_ds = ray.train.get_dataset_shard("train")
    
    for epoch in range(num_epochs):
        # Iterate with batches — streaming, memory-efficient
        for batch in train_ds.iter_torch_batches(batch_size=32, device="cuda"):
            loss = model(batch["input_ids"], batch["labels"])
            ...
```

## Ray Tune Integration

```python
from ray import tune

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
)

tuner = tune.Tuner(
    trainer,
    param_space={
        "train_loop_config": {
            "lr": tune.loguniform(1e-5, 1e-3),
            "batch_size": tune.choice([16, 32, 64]),
            "epochs": 3,
        },
    },
    tune_config=tune.TuneConfig(
        num_samples=10,
        metric="eval_loss",
        mode="min",
        scheduler=tune.schedulers.ASHAScheduler(max_t=3, grace_period=1),
    ),
)
results = tuner.fit()
```

## Multi-Node Training on Kubernetes

Deploy as a RayJob with `shutdownAfterJobFinishes: true`:

```yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: distributed-training
spec:
  entrypoint: python train.py
  shutdownAfterJobFinishes: true
  activeDeadlineSeconds: 86400
  submissionMode: K8sJobMode
  rayClusterSpec:
    headGroupSpec:
      template:
        spec:
          containers:
          - name: ray-head
            resources:
              limits:
                cpu: "4"
                memory: 16Gi
    workerGroupSpecs:
    - groupName: gpu-workers
      replicas: 8
      template:
        spec:
          containers:
          - name: ray-worker
            resources:
              limits:
                cpu: "8"
                memory: 64Gi
                nvidia.com/gpu: "1"
```

### Shared Storage for Checkpoints

Mount a shared PVC or use S3/GCS:
```python
run_config = RunConfig(
    storage_path="/mnt/shared-checkpoints",   # PVC mounted on all pods
    # or: storage_path="s3://bucket/ray-results",
)
```

## Debugging

For communication errors, OOM, slow data loading, and common training issues, see `references/troubleshooting.md`.

## Cross-References

- [ray-core](../ray-core/) — Ray tasks, actors, and object store fundamentals
- [ray-data](../ray-data/) — Streaming data pipelines for training
- [aws-efa](../aws-efa/) — EFA networking for multi-node Ray Train on EKS
- [aws-fsx](../aws-fsx/) — FSx storage for training data and checkpoints
- [kuberay](../kuberay/) — Deploy training jobs on Kubernetes via RayJob CRD
- [pytorch](../pytorch/) — PyTorch distributed training concepts
- [fsdp](../fsdp/) — FSDP for model parallelism within Ray Train

## Reference

- [Ray Train docs](https://docs.ray.io/en/latest/train/train.html)
- [TorchTrainer API](https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.TorchTrainer.html)
- [Ray Train examples](https://docs.ray.io/en/latest/train/examples.html)
- `references/troubleshooting.md` — common errors and fixes
- `assets/architecture.md` — Mermaid architecture diagrams
