---
name: ray-train
description: >
  Scale distributed model training with Ray Train across GPUs and nodes. Use when:
  (1) Setting up distributed training (TorchTrainer, data-parallel scaling),
  (2) Configuring ScalingConfig (num_workers, use_gpu, resources_per_worker),
  (3) Implementing checkpointing and fault tolerance (ray.train.report, Checkpoint),
  (4) Fine-tuning HuggingFace models with Ray Train (Transformers Trainer integration),
  (5) Using DeepSpeed or FSDP for large model training (ZeRO stages, sharding),
  (6) Feeding data from Ray Data into training (streaming datasets),
  (7) Running hyperparameter tuning with Ray Tune,
  (8) Debugging training issues (communication errors, OOM, slow data loading, rank mismatches).
---

# Ray Train

Distributed training library for scaling model training across GPUs and nodes with minimal code changes.

**Docs:** https://docs.ray.io/en/latest/train/train.html
**Version:** Ray 2.53.0

## Supported Frameworks

| Framework | Integration |
|---|---|
| PyTorch | `TorchTrainer` + `prepare_model` / `prepare_data_loader` |
| HuggingFace Transformers | `TorchTrainer` + `prepare_trainer` + `RayTrainReportCallback` |
| PyTorch Lightning | `TorchTrainer` + `RayTrainReportCallback` |
| HuggingFace Accelerate | `TorchTrainer` + Accelerate API |
| DeepSpeed | `TorchTrainer` + `deepspeed.initialize` |
| XGBoost / LightGBM | `XGBoostTrainer` / `LightGBMTrainer` |

All deep learning frameworks use `TorchTrainer` as the launcher.

## PyTorch Training (Quick Start)

Three changes to convert single-GPU PyTorch code to distributed:

```python
import os
import tempfile
import torch
import ray.train
import ray.train.torch
from ray.train import ScalingConfig, Checkpoint
from ray.train.torch import TorchTrainer

def train_func(config):
    # 1. Prepare model (wraps in DDP, moves to correct device)
    model = build_model()
    model = ray.train.torch.prepare_model(model)

    # 2. Prepare dataloader (adds DistributedSampler, moves to device)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    train_loader = ray.train.torch.prepare_data_loader(train_loader)

    for epoch in range(config["epochs"]):
        if ray.train.get_context().get_world_size() > 1:
            train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            loss = train_step(model, batch)

        # 3. Report metrics + checkpoint (only from rank 0 for DDP)
        metrics = {"loss": loss.item(), "epoch": epoch}
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = None
            if ray.train.get_context().get_world_rank() == 0:
                torch.save(model.module.state_dict(), os.path.join(tmpdir, "model.pt"))
                checkpoint = Checkpoint.from_directory(tmpdir)
            ray.train.report(metrics, checkpoint=checkpoint)

# Launch distributed training
trainer = TorchTrainer(
    train_func,
    train_loop_config={"epochs": 10},
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
    # run_config=ray.train.RunConfig(storage_path="s3://bucket/checkpoints"),
)
result = trainer.fit()

# Load trained model
with result.checkpoint.as_directory() as ckpt_dir:
    state_dict = torch.load(os.path.join(ckpt_dir, "model.pt"))
```

### Key Utilities

| Function | Purpose |
|---|---|
| `ray.train.torch.prepare_model(model)` | Wrap in DDP, move to GPU |
| `ray.train.torch.prepare_data_loader(loader)` | Add DistributedSampler, move batches to device |
| `ray.train.report(metrics, checkpoint=)` | Report metrics and save checkpoint |
| `ray.train.get_context()` | Get world_rank, world_size, local_rank |
| `ray.train.get_dataset_shard("train")` | Get Ray Data shard for this worker |

**Important:** `prepare_model` returns a DDP-wrapped model. Save `model.module.state_dict()` (not `model.state_dict()`) to avoid the `module.` prefix in keys.

## ScalingConfig

```python
from ray.train import ScalingConfig

scaling_config = ScalingConfig(
    num_workers=4,                    # number of training workers
    use_gpu=True,                     # allocate 1 GPU per worker
    resources_per_worker={            # custom resources per worker
        "CPU": 4,
        "GPU": 1,
    },
    trainer_resources={"CPU": 1},     # resources for the driver process
)
```

**Batch size math:**
```
global_batch_size = worker_batch_size × num_workers
```

## Checkpointing & Fault Tolerance

### Checkpoint Strategy

```python
from ray.train import RunConfig, CheckpointConfig

run_config = RunConfig(
    storage_path="s3://bucket/experiments",   # persistent storage (required for multi-node)
    name="my-experiment",
    checkpoint_config=CheckpointConfig(
        num_to_keep=3,                        # keep top N checkpoints
        checkpoint_score_attribute="eval_loss",
        checkpoint_score_order="min",         # "min" or "max"
    ),
)
```

### Fault Tolerance

```python
from ray.train import FailureConfig

run_config = RunConfig(
    failure_config=FailureConfig(
        max_failures=3,           # auto-restart on crash, up to 3 times
    ),
)
```

Workers resume from the latest checkpoint. Your training function must handle checkpoint restoration:

```python
def train_func(config):
    checkpoint = ray.train.get_checkpoint()
    start_epoch = 0
    if checkpoint:
        with checkpoint.as_directory() as ckpt_dir:
            state_dict = torch.load(os.path.join(ckpt_dir, "model.pt"))
            model.load_state_dict(state_dict)
            start_epoch = torch.load(os.path.join(ckpt_dir, "meta.pt"))["epoch"] + 1

    for epoch in range(start_epoch, config["epochs"]):
        # ... training loop with checkpoint saving
```

## HuggingFace Transformers Integration

Minimal changes to any HuggingFace `Trainer` script:

```python
import ray.train.huggingface.transformers
from ray.train.torch import TorchTrainer

def train_func():
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    dataset = load_dataset("yelp_review_full")

    training_args = TrainingArguments(
        output_dir="output",
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=dataset["train"], eval_dataset=dataset["test"],
    )

    # Two lines to add:
    callback = ray.train.huggingface.transformers.RayTrainReportCallback()
    trainer.add_callback(callback)
    trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)

    trainer.train()

ray_trainer = TorchTrainer(
    train_func,
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
)
result = ray_trainer.fit()
```

## DeepSpeed Integration

```python
import deepspeed
from ray.train.torch import TorchTrainer

def train_func(config):
    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    # DeepSpeed config
    ds_config = {
        "train_batch_size": 64 * ray.train.get_context().get_world_size(),
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,                     # ZeRO Stage 2
            "offload_optimizer": {"device": "cpu"},
        },
        "gradient_accumulation_steps": 4,
    }

    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
    )

    for batch in train_loader:
        loss = model(batch)
        model.backward(loss)
        model.step()

trainer = TorchTrainer(
    train_func,
    train_loop_config={"lr": 1e-4},
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
)
```

**ZeRO Stages:** Stage 1 (optimizer partitioning), Stage 2 (+ gradient partitioning), Stage 3 (+ parameter partitioning). Higher stages save more memory but add communication overhead.

## FSDP Integration

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def train_func(config):
    model = build_model()

    # Don't call prepare_model — use FSDP directly
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
        ),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    for batch in train_loader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

trainer = TorchTrainer(
    train_func,
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
    torch_config=ray.train.torch.TorchConfig(backend="nccl"),
)
```

**Note:** When using FSDP or DeepSpeed, do NOT call `ray.train.torch.prepare_model()` — it wraps in DDP which conflicts with these frameworks. See the **fsdp** skill for detailed FSDP configuration.

## Data Loading with Ray Data

```python
import ray

ds = ray.data.read_parquet("s3://bucket/training-data/")

def train_func(config):
    train_shard = ray.train.get_dataset_shard("train")

    for epoch in range(config["epochs"]):
        for batch in train_shard.iter_torch_batches(batch_size=64):
            # batch is Dict[str, torch.Tensor]
            loss = train_step(model, batch)

trainer = TorchTrainer(
    train_func,
    train_loop_config={"epochs": 10},
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
    datasets={"train": ds},
)
```

Benefits: streaming (no full materialization), automatic sharding across workers, CPU preprocessing offloaded from GPU workers.

## Ray Tune Integration

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

trainer = TorchTrainer(
    train_func,
    train_loop_config={"lr": 1e-4, "epochs": 10},
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
)

tuner = tune.Tuner(
    trainer,
    param_space={
        "train_loop_config": {
            "lr": tune.loguniform(1e-5, 1e-2),
            "epochs": 10,
        }
    },
    tune_config=tune.TuneConfig(
        num_samples=10,
        scheduler=ASHAScheduler(metric="eval_loss", mode="min"),
    ),
)
results = tuner.fit()
best_result = results.get_best_result(metric="eval_loss", mode="min")
```

## TorchConfig

```python
from ray.train.torch import TorchConfig

torch_config = TorchConfig(
    backend="nccl",        # "nccl" for GPU (default), "gloo" for CPU
    timeout_s=1800,        # distributed timeout (seconds)
)

trainer = TorchTrainer(
    train_func,
    torch_config=torch_config,
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
)
```

## Multi-Node Training

For multi-node, checkpoint storage must be shared/remote:

```python
run_config = RunConfig(
    storage_path="s3://bucket/experiments",  # or NFS path
    name="multi-node-experiment",
)
```

Ray Train handles `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`, and `LOCAL_RANK` env vars automatically.

## Debugging

For communication errors, OOM, slow data loading, and common training issues, see `references/troubleshooting.md`.
