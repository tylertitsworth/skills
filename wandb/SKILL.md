---
name: wandb
description: >
  Track ML experiments, version artifacts, and manage models with Weights & Biases (W&B).
  Use when: (1) Setting up wandb experiment tracking (init, config, logging metrics/media),
  (2) Integrating with PyTorch, HuggingFace, Lightning, or Ray Train,
  (3) Versioning datasets and models with Artifacts, (4) Using the Model Registry for
  staging and deployment, (5) Running hyperparameter sweeps (Bayesian, grid, random),
  (6) Querying runs and exporting data via the Public API,
  (7) Advanced operations (alerts, custom charts, Tables, reports, run grouping),
  (8) Debugging tracking issues and run recovery.
---

# Weights & Biases (wandb)

W&B is the ML experiment tracking, artifact versioning, and model registry platform. Assumes a dedicated W&B deployment with BYOB (Bring Your Own Bucket). SDK version: **0.18.x+**.

## Setup

Add `wandb` to your container image dependencies. Configure via env vars in your pod spec:

```yaml
env:
- name: WANDB_API_KEY
  valueFrom:
    secretKeyRef:
      name: wandb-secret
      key: api-key
- name: WANDB_BASE_URL
  value: "https://wandb.example.com"    # your dedicated deployment
- name: WANDB_PROJECT
  value: "my-project"
- name: WANDB_ENTITY
  value: "my-team"
```

### Environment Variables Reference

| Variable | Purpose | Default |
|---|---|---|
| `WANDB_API_KEY` | Authentication key | required |
| `WANDB_BASE_URL` | W&B server URL | `https://api.wandb.ai` |
| `WANDB_PROJECT` | Default project name | `"uncategorized"` |
| `WANDB_ENTITY` | Team or user entity | User default |
| `WANDB_RUN_GROUP` | Group related runs | None |
| `WANDB_JOB_TYPE` | Run type label (train, eval, etc.) | None |
| `WANDB_TAGS` | Comma-separated tags | None |
| `WANDB_NOTES` | Run description | None |
| `WANDB_NAME` | Run display name | Auto-generated |
| `WANDB_DIR` | Local directory for run files | `./wandb` |
| `WANDB_SILENT` | Suppress console output | `false` |
| `WANDB_RESUME` | Resume behavior | `"never"` |
| `WANDB_LOG_MODEL` | Auto-log model checkpoints | `false` |
| `WANDB_WATCH` | Auto-log model gradients/parameters | `false` |
| `WANDB_DISABLED` | Disable wandb entirely | `false` |

## Experiment Tracking

### wandb.init() Settings

```python
import wandb

run = wandb.init(
    project="my-project",
    entity="my-team",
    name="llama-8b-sft-lr1e4",
    group="llama-8b-sft",                # group related runs
    job_type="train",                     # train, eval, preprocess, etc.
    tags=["sft", "llama", "8b"],
    notes="SFT with learning rate 1e-4",
    config={                              # hyperparameters and settings
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "learning_rate": 1e-4,
        "batch_size": 32,
        "epochs": 3,
    },
    resume="allow",                       # resume behavior (see below)
    reinit=True,                          # allow multiple init() in same process
    save_code=True,                       # save the main script
    settings=wandb.Settings(
        start_method="thread",            # thread, fork, forkserver
    ),
)
```

### Resume Behavior

| Value | Behavior |
|---|---|
| `"never"` (default) | Always create new run |
| `"allow"` | Resume if run ID exists, else create new |
| `"must"` | Must resume existing run (error if not found) |
| `"auto"` | Auto-resume from environment (uses `WANDB_RUN_ID`) |

For crash recovery in training jobs, set `resume="allow"` and provide a deterministic `id`:

```python
run = wandb.init(
    id=f"training-{model_name}-{experiment_id}",  # deterministic ID
    resume="allow",
    project="my-project",
)
```

### Logging Metrics

```python
# Basic logging
wandb.log({"train/loss": 0.5, "train/accuracy": 0.85, "epoch": 1})

# Step-based logging (explicit step)
wandb.log({"train/loss": 0.3}, step=100)

# Commit control (batch multiple calls into one step)
wandb.log({"train/loss": 0.3}, commit=False)
wandb.log({"train/accuracy": 0.9}, commit=True)  # both logged at same step

# Summary metrics (final values shown in run table)
wandb.run.summary["best_accuracy"] = 0.95
wandb.run.summary["best_epoch"] = 7

# Define metric properties (x-axis, summary, goal)
wandb.define_metric("train/*", step_metric="global_step")
wandb.define_metric("eval/*", step_metric="epoch")
wandb.define_metric("eval/accuracy", summary="max", goal="maximize")
wandb.define_metric("eval/loss", summary="min", goal="minimize")
```

### define_metric Reference

| Setting | Purpose | Values |
|---|---|---|
| `step_metric` | X-axis for this metric | Any logged metric name |
| `summary` | How to summarize in run table | `"min"`, `"max"`, `"mean"`, `"last"`, `"best"`, `"copy"`, `"none"` |
| `goal` | Optimization direction | `"minimize"`, `"maximize"` |
| `hidden` | Hide from default charts | `True`/`False` |
| `overwrite` | Allow redefining | `True`/`False` |

### Logging Media

```python
# Images
wandb.log({"samples": [wandb.Image(img, caption=f"Sample {i}") for i, img in enumerate(images)]})

# Tables (structured data)
table = wandb.Table(columns=["input", "prediction", "label", "correct"])
for inp, pred, label in results:
    table.add_data(inp, pred, label, pred == label)
wandb.log({"predictions": table})

# Histograms
wandb.log({"weight_dist": wandb.Histogram(model.fc.weight.data.cpu())})

# Audio
wandb.log({"audio": wandb.Audio(audio_array, sample_rate=16000)})

# Point clouds, 3D objects, HTML
wandb.log({"scene": wandb.Object3D(point_cloud)})
wandb.log({"report": wandb.Html(html_string)})
```

### Custom Charts

Built-in chart functions: `wandb.plot.line()`, `wandb.plot.scatter()`, `wandb.plot.bar()`, `wandb.plot.confusion_matrix()`, `wandb.plot.pr_curve()`, `wandb.plot.roc_curve()`. All take a `wandb.Table` and column names.

### Alerts

```python
# Send alert when metric crosses threshold
if val_loss > 2.0:
    wandb.alert(
        title="Training diverging",
        text=f"val_loss={val_loss:.4f} exceeded threshold at step {step}",
        level=wandb.AlertLevel.WARN,  # INFO, WARN, ERROR
        wait_duration=300,            # don't re-alert for 5 minutes
    )
```

## Framework Integrations

### PyTorch (Manual)

```python
wandb.watch(model, log="all", log_freq=100)  # log gradients + parameters

for step, batch in enumerate(dataloader):
    loss = train_step(model, batch)
    wandb.log({"train/loss": loss, "global_step": step})

wandb.unwatch(model)
```

`wandb.watch()` options:

| Setting | Purpose | Default |
|---|---|---|
| `log` | What to log: `"gradients"`, `"parameters"`, `"all"`, `None` | `"gradients"` |
| `log_freq` | Log every N steps | `1000` |
| `log_graph` | Log computation graph | `False` |

### HuggingFace Transformers

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    report_to="wandb",
    run_name="llama-sft",
    logging_steps=10,
    # WANDB_PROJECT, WANDB_ENTITY set via env vars in pod spec
)
```

Set `WANDB_LOG_MODEL="checkpoint"` to auto-log checkpoints as artifacts, or `"end"` to log only the final model.

### PyTorch Lightning

```python
from lightning.pytorch.loggers import WandbLogger

logger = WandbLogger(
    project="my-project",
    name="my-run",
    log_model="all",           # log checkpoints as artifacts
    save_dir="/tmp/wandb",
)
trainer = pl.Trainer(logger=logger)
```

### Ray Train

```python
from ray.train import RunConfig
from ray.air.integrations.wandb import WandbLoggerCallback

run_config = RunConfig(
    callbacks=[WandbLoggerCallback(
        project="my-project",
        group="ray-experiment",
        log_config=True,
    )],
)
```

### lm-evaluation-harness

```python
import lm_eval
results = lm_eval.simple_evaluate(
    model="vllm",
    model_args="pretrained=my-model",
    tasks=["mmlu"],
    wandb_args="project=eval,name=llama-8b-mmlu,job_type=eval",
)
```

## Artifacts

Artifacts version datasets, models, and other files with automatic lineage tracking.

### Artifact Types

| Type | Convention | Use Case |
|---|---|---|
| `dataset` | Versioned training/eval data | `my-dataset:v3` |
| `model` | Model checkpoints | `llama-sft:latest` |
| `code` | Source code snapshots | `training-code:v1` |
| Custom string | Any file collection | `configs:v2` |

### Creating and Logging Artifacts

```python
# Log a model checkpoint
artifact = wandb.Artifact(
    name="llama-sft-model",
    type="model",
    description="SFT fine-tuned Llama 3.1 8B",
    metadata={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "epochs": 3,
        "eval_loss": 0.42,
    },
)
artifact.add_dir("./checkpoint-best")           # add entire directory
# or: artifact.add_file("model.safetensors")    # add single file
# or: artifact.add_reference("s3://bucket/model/")  # reference (no copy)
wandb.log_artifact(artifact)
```

### Artifact References (BYOB)

With BYOB, artifacts can reference files in your own bucket without copying:

```python
artifact = wandb.Artifact("training-data", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/openorca/", max_objects=100000)
wandb.log_artifact(artifact)

# Later, download (fetches from your bucket)
artifact = wandb.use_artifact("training-data:latest")
artifact.download(root="/data/openorca")
```

### Using Artifacts (Input)

```python
# Declare dependency and download
artifact = wandb.use_artifact("my-team/my-project/training-data:v3")
data_dir = artifact.download(root="/data/training")

# Get metadata without downloading
artifact = wandb.use_artifact("my-team/my-project/llama-sft-model:latest")
print(artifact.metadata)  # {"model": "...", "epochs": 3, ...}
```

### Artifact Aliases

| Alias | Purpose |
|---|---|
| `latest` | Most recently logged version |
| `best` | Custom alias for best-performing version |
| `production` | Custom alias for deployed version |

```python
# Set aliases
artifact.aliases = ["best", "production"]
artifact.save()

# Or via API
api = wandb.Api()
artifact = api.artifact("my-team/my-project/llama-sft-model:v5")
artifact.aliases.append("production")
artifact.save()
```

## Model Registry

The Model Registry organizes model artifacts with lifecycle management:

```python
# Link artifact to registry
run.link_artifact(
    artifact,
    target_path="my-team/model-registry/LLama-3.1-SFT",
    aliases=["staging"],
)
```

### Registry Lifecycle

| Stage | Meaning |
|---|---|
| `staging` | Ready for validation |
| `production` | Deployed/serving |
| Custom aliases | Any workflow-specific labels |

### Querying the Registry

```python
api = wandb.Api()

# List all registered models
collections = api.artifact_type("model", project="model-registry").collections()
for collection in collections:
    print(collection.name, collection.aliases)

# Get production model
model = api.artifact("my-team/model-registry/LLama-3.1-SFT:production")
model.download(root="/models/production")
```

## Sweeps (Hyperparameter Optimization)

### Sweep Config

```python
sweep_config = {
    "method": "bayes",                    # bayes, grid, random
    "metric": {"name": "eval/accuracy", "goal": "maximize"},
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-3,
        },
        "batch_size": {"values": [16, 32, 64]},
        "epochs": {"value": 10},          # fixed parameter
        "optimizer": {"values": ["adam", "adamw"]},
        "weight_decay": {"min": 0.0, "max": 0.1},
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 3,
        "eta": 2,
    },
}
```

### Sweep Parameter Distributions

| Distribution | Fields | For |
|---|---|---|
| `constant` | `value` | Fixed value |
| `categorical` | `values` (list) | Discrete choices |
| `uniform` | `min`, `max` | Continuous uniform |
| `log_uniform_values` | `min`, `max` | Log-uniform (for learning rates) |
| `normal` | `mu`, `sigma` | Normal distribution |
| `int_uniform` | `min`, `max` | Integer range |
| `q_uniform` | `min`, `max`, `q` | Quantized uniform |

### Running Sweeps

```python
sweep_id = wandb.sweep(sweep_config, project="hp-search")
wandb.agent(sweep_id, function=train_func, count=50)
```

### Sweep Methods

| Method | Description | Best For |
|---|---|---|
| `bayes` | Bayesian optimization with Gaussian processes | Small-medium search spaces |
| `grid` | Exhaustive grid search | Small discrete spaces |
| `random` | Random sampling | Large spaces, baseline |

## Public API (Programmatic Access)

```python
api = wandb.Api()

# Query runs with filters
runs = api.runs("my-team/my-project", filters={
    "tags": {"$in": ["production"]},
    "summary_metrics.eval/accuracy": {"$gt": 0.9},
    "state": "finished",
})

# Export to DataFrame
import pandas as pd
data = [{
    "name": r.name,
    "lr": r.config.get("learning_rate"),
    "accuracy": r.summary.get("eval/accuracy"),
    "duration": r.summary.get("_wandb", {}).get("runtime"),
} for r in runs]
df = pd.DataFrame(data)

# Download run history
run = api.run("my-team/my-project/run-id")
history = run.history(samples=1000)       # pandas DataFrame
system_metrics = run.history(stream="events")  # system metrics (GPU, CPU)

# Delete old runs
for run in api.runs("my-team/my-project", filters={"created_at": {"$lt": "2024-01-01"}}):
    run.delete()
```

### API Filter Operators

| Operator | Example |
|---|---|
| `$eq` | `{"config.model": {"$eq": "llama"}}` |
| `$ne` | `{"state": {"$ne": "crashed"}}` |
| `$gt`, `$gte`, `$lt`, `$lte` | `{"summary_metrics.loss": {"$lt": 0.5}}` |
| `$in` | `{"tags": {"$in": ["prod"]}}` |
| `$nin` | `{"tags": {"$nin": ["debug"]}}` |
| `$regex` | `{"name": {"$regex": "llama.*"}}` |

## Run Grouping and Comparison

### Groups

Group related runs (e.g., distributed training workers, experiment variants):

```python
wandb.init(group="llama-8b-ablation", job_type="train")
```

All runs in a group appear together in the UI with aggregated metrics.

### Tags

```python
# Set at init
wandb.init(tags=["sft", "llama-8b", "production"])

# Add/remove during run
wandb.run.tags = wandb.run.tags + ("best-model",)
```

## Debugging

See `references/troubleshooting.md` for:
- Sync failures and recovery
- Large artifact handling
- Rate limiting
- Run resume and crash recovery
- Common integration pitfalls

## Cross-References

- [pytorch](../pytorch/) — PyTorch training loops with W&B logging
- [ray-train](../ray-train/) — Ray Train experiment tracking integration
- [huggingface-transformers](../huggingface-transformers/) — HF Trainer with W&B callback
- [hydra](../hydra/) — Hydra config management for experiment parameters
- [llm-evaluation](../llm-evaluation/) — Log evaluation results to W&B

## Reference

- [W&B Documentation](https://docs.wandb.ai/)
- [wandb GitHub](https://github.com/wandb/wandb)
- [Python SDK Reference](https://docs.wandb.ai/models/ref/python)
- [Public API Reference](https://docs.wandb.ai/ref/python/public-api/)
- `references/troubleshooting.md` — common errors and fixes
- `scripts/sweep_agent.py` — W&B Sweep agent for Bayesian hyperparameter tuning with early termination
