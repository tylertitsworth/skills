---
name: wandb
description: >
  Track ML experiments, version artifacts, and manage models with Weights & Biases (W&B).
  Use when: (1) Setting up wandb experiment tracking (init, config, logging), (2) Integrating
  with PyTorch, HuggingFace, Lightning, or Ray Train, (3) Versioning datasets and models with
  Artifacts, (4) Using the Model Registry for staging and deployment, (5) Running hyperparameter
  sweeps, (6) Building reports and dashboards, (7) Querying runs programmatically via the API,
  (8) Self-hosting W&B Server on Kubernetes, (9) Debugging sync failures or offline mode issues.
---

# Weights & Biases (wandb)

W&B is the ML experiment tracking, artifact versioning, and model registry platform. SDK version: **0.18.x+**.

## Setup

Add `wandb` to your container image dependencies. Configure via env vars in your pod spec:

```yaml
env:
- name: WANDB_API_KEY
  valueFrom:
    secretKeyRef:
      name: wandb-secret
      key: api-key
- name: WANDB_PROJECT
  value: "my-project"
- name: WANDB_ENTITY
  value: "my-team"
- name: WANDB_BASE_URL          # for self-hosted W&B
  value: "https://wandb.example.com"
- name: WANDB_MODE              # "offline" for air-gapped clusters
  value: "online"
```

## Experiment Tracking

### Initialize a Run

```python
import wandb

with wandb.init(
    project="llm-finetune",
    entity="ml-team",           # team or username
    name="bert-lr-sweep-01",    # human-readable run name
    tags=["bert", "finetune", "a100"],
    config={                    # hyperparameters
        "learning_rate": 2e-5,
        "batch_size": 32,
        "epochs": 10,
        "model": "bert-base-uncased",
        "gpu": "A100",
    },
    job_type="training",        # groups by purpose in UI
    notes="Testing LR sweep on A100 node",
) as run:
    # Training loop
    for epoch in range(run.config.epochs):
        train_loss, val_loss, val_acc = train_epoch(model, epoch)

        run.log({
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "epoch": epoch,
        })

    # Log final metrics
    run.summary["best_accuracy"] = best_acc
    run.summary["total_params"] = count_params(model)
```

### Config Management

```python
# Update config after init
run.config.update({"optimizer": "adamw", "warmup_steps": 500})

# Config from argparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-4)
args = parser.parse_args()
run.config.update(args)

# Config from a YAML/dict file
run.config.update(yaml.safe_load(open("config.yaml")))
```

### Logging Options

```python
# Metrics at a step
run.log({"loss": 0.5, "gpu_mem_gb": 38.2}, step=100)

# Histograms and distributions
run.log({"gradients": wandb.Histogram(grad_norms)})

# Images
run.log({"predictions": [wandb.Image(img, caption=f"pred: {p}") for img, p in preds]})

# Tables (structured data)
table = wandb.Table(columns=["input", "prediction", "label"])
for row in eval_results:
    table.add_data(row["input"], row["pred"], row["label"])
run.log({"eval_results": table})

# Custom charts
run.log({"pr_curve": wandb.plot.pr_curve(y_true, y_scores, labels=class_names)})
run.log({"conf_mat": wandb.plot.confusion_matrix(y_true, y_pred, class_names)})

# System metrics are logged automatically (GPU util, memory, CPU, etc.)
```

### Groups and Job Types

```python
# Group related runs (e.g., distributed training workers)
wandb.init(group="ddp-experiment-1", job_type="train-worker")

# Group cross-validation folds
for fold in range(5):
    with wandb.init(group="cv-experiment", job_type=f"fold-{fold}") as run:
        ...
```

## Framework Integrations

### PyTorch (Manual)

```python
with wandb.init(project="pytorch-training", config=config) as run:
    # Watch model — logs gradients, parameters
    run.watch(model, log="all", log_freq=100)

    for epoch in range(config["epochs"]):
        for batch in dataloader:
            loss = train_step(model, batch)
            run.log({"train/loss": loss.item()})

    # Save model checkpoint
    torch.save(model.state_dict(), "model.pt")
    artifact = wandb.Artifact("trained-model", type="model")
    artifact.add_file("model.pt")
    run.log_artifact(artifact)
```

### HuggingFace Transformers

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    report_to="wandb",               # enables W&B integration
    run_name="hf-bert-finetune",
    logging_steps=10,
    num_train_epochs=3,
    per_device_train_batch_size=16,
)

# wandb.init() is called automatically by Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)
trainer.train()
```

**Container env vars:**
```yaml
- name: WANDB_PROJECT
  value: "hf-experiments"
- name: WANDB_LOG_MODEL
  value: "checkpoint"  # log model checkpoints as artifacts
```

### PyTorch Lightning

```python
from lightning.pytorch.loggers import WandbLogger

wandb_logger = WandbLogger(
    project="lightning-training",
    name="resnet50-run",
    log_model="all",  # log checkpoints as artifacts
)

trainer = L.Trainer(
    max_epochs=10,
    logger=wandb_logger,
    accelerator="gpu",
    devices=1,
)
trainer.fit(model, train_dataloader, val_dataloader)
```

### Ray Train

```python
from ray.train import RunConfig
from ray.air.integrations.wandb import WandbLoggerCallback

run_config = RunConfig(
    name="ray-train-experiment",
    callbacks=[
        WandbLoggerCallback(
            project="ray-experiments",
            api_key_file="~/.wandb_key",  # or set WANDB_API_KEY
        )
    ],
)

# Use with TorchTrainer, etc.
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    run_config=run_config,
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
)
```

## Artifacts

Version and track datasets, models, and any files:

### Log an Artifact

```python
with wandb.init(project="my-project", job_type="data-prep") as run:
    artifact = wandb.Artifact(
        name="training-dataset",
        type="dataset",
        description="Preprocessed training data v2",
        metadata={"num_samples": 50000, "source": "s3://data/raw"},
    )
    # Add files or directories
    artifact.add_file("data/train.parquet")
    artifact.add_dir("data/images/")
    # Or add a reference (pointer, no upload)
    artifact.add_reference("s3://bucket/large-dataset/")

    run.log_artifact(artifact)
```

### Use an Artifact

```python
with wandb.init(project="my-project", job_type="training") as run:
    # Download and use
    artifact = run.use_artifact("training-dataset:latest")
    data_dir = artifact.download()  # downloads to local cache

    # Or use a specific version
    artifact = run.use_artifact("training-dataset:v3")

    # Or use by alias
    artifact = run.use_artifact("training-dataset:best")
```

### Model Checkpoints as Artifacts

```python
with wandb.init(project="training") as run:
    for epoch in range(epochs):
        train(model, epoch)
        # Save checkpoint every N epochs
        if epoch % 5 == 0:
            artifact = wandb.Artifact(
                f"model-checkpoint",
                type="model",
                metadata={"epoch": epoch, "val_loss": val_loss},
            )
            torch.save(model.state_dict(), "checkpoint.pt")
            artifact.add_file("checkpoint.pt")
            run.log_artifact(artifact)  # creates new version automatically
```

## Model Registry

Promote artifacts to a central registry for team-wide access:

```python
with wandb.init(project="training") as run:
    # Log a model artifact
    logged_artifact = run.log_artifact(
        artifact_or_path="./model_weights/",
        name="llama-finetuned",
        type="model",
    )

    # Link to registry collection
    run.link_artifact(
        artifact=logged_artifact,
        target_path="wandb-registry-model/production-models",
    )
```

**Download from registry:**
```python
with wandb.init() as run:
    artifact = run.use_artifact(
        "wandb-registry-model/production-models:latest"
    )
    model_dir = artifact.download()
```

**Aliases for lifecycle management:**
```python
# Promote via API
api = wandb.Api()
artifact = api.artifact("my-team/my-project/llama-finetuned:v5")
artifact.aliases.append("staging")
artifact.save()

# Later, promote to production
artifact.aliases.append("production")
artifact.aliases.remove("staging")
artifact.save()
```

## Sweeps (Hyperparameter Optimization)

### Define a Sweep

```python
sweep_config = {
    "method": "bayes",  # bayes, grid, random
    "metric": {"name": "val/accuracy", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"min": 1e-6, "max": 1e-3, "distribution": "log_uniform_values"},
        "batch_size": {"values": [16, 32, 64]},
        "epochs": {"value": 10},  # fixed
        "optimizer": {"values": ["adam", "adamw", "sgd"]},
        "weight_decay": {"min": 0.0, "max": 0.1},
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 3,
        "eta": 2,
    },
}
```

### Run a Sweep

```python
def train_sweep():
    with wandb.init() as run:
        config = run.config
        model = build_model(config)
        for epoch in range(config.epochs):
            loss, acc = train_epoch(model, config)
            run.log({"val/accuracy": acc, "val/loss": loss})

# Create and run sweep
sweep_id = wandb.sweep(sweep_config, project="hp-search")
wandb.agent(sweep_id, function=train_sweep, count=50)  # max 50 runs
```

### Sweep from CLI

```bash
# sweep.yaml contains the config
wandb sweep sweep.yaml
# Returns: wandb agent my-team/my-project/<sweep-id>

# Run agents (can run on multiple machines/GPUs)
CUDA_VISIBLE_DEVICES=0 wandb agent my-team/my-project/<sweep-id> &
CUDA_VISIBLE_DEVICES=1 wandb agent my-team/my-project/<sweep-id> &
```

## Public API (Programmatic Access)

```python
api = wandb.Api()

# Query runs
runs = api.runs("my-team/my-project", filters={"tags": "production"})
for run in runs:
    print(f"{run.name}: {run.summary.get('val/accuracy', 'N/A')}")

# Get a specific run
run = api.run("my-team/my-project/run-id")
print(run.config)
print(run.summary)

# Download run history as DataFrame
history = run.history()  # pandas DataFrame

# Export metrics for comparison
import pandas as pd
data = []
for run in api.runs("my-team/my-project"):
    data.append({
        "name": run.name,
        "lr": run.config.get("learning_rate"),
        "accuracy": run.summary.get("val/accuracy"),
    })
df = pd.DataFrame(data)

# Download artifacts
artifact = api.artifact("my-team/my-project/model:best")
artifact.download(root="./downloaded_model")

# Delete runs programmatically
run = api.run("my-team/my-project/old-run-id")
run.delete()
```

## Self-Hosted W&B Server

For homelab/on-prem deployments using the W&B Kubernetes Operator:

```bash
# Add the W&B Helm repo
helm repo add wandb https://charts.wandb.ai
helm repo update

# Install the operator
helm install wandb-operator wandb/operator \
  --namespace wandb --create-namespace

# Create a values file for the W&B instance
cat <<EOF > wandb-values.yaml
apiVersion: apps.wandb.com/v1
kind: WeightsAndBiases
metadata:
  name: wandb
  namespace: wandb
spec:
  values:
    global:
      host: https://wandb.example.com
      license: "YOUR_LICENSE_KEY"
      bucket:
        provider: s3  # or gcs, az
        name: wandb-artifacts
        region: us-east-1
    ingress:
      enabled: true
      class: nginx
    mysql:
      host: mysql.wandb.svc
      database: wandb_local
      user: wandb
      password: <secret>
EOF

kubectl apply -f wandb-values.yaml
```

**Point clients at self-hosted server** — set `WANDB_BASE_URL` env var in pod specs:

## Debugging

See `references/troubleshooting.md` for:
- Offline mode and sync failures
- Large artifact handling
- Rate limiting and network issues
- Run resume and crash recovery
- Common integration pitfalls

## Reference

- [W&B Documentation](https://docs.wandb.ai/)
- [wandb GitHub](https://github.com/wandb/wandb)
- [W&B Helm Charts](https://github.com/wandb/helm-charts)
- [Python SDK Reference](https://docs.wandb.ai/models/ref/python)
- `references/troubleshooting.md` — common errors and fixes
