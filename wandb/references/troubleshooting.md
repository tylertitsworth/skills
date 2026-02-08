# W&B Troubleshooting

## Offline Mode and Syncing

### Working offline

When no network is available (air-gapped cluster, train on a node without internet):

```bash
# Set offline mode
export WANDB_MODE="offline"

# Or per-run
wandb.init(mode="offline")
```

Runs are saved to `./wandb/offline-run-*` directories.

### Syncing offline runs

```bash
# Sync a single run
wandb sync ./wandb/offline-run-20240101_120000-abc123

# Sync all offline runs in a directory
wandb sync --sync-all ./wandb/

# Sync to a specific project
wandb sync --project my-project ./wandb/offline-run-*
```

### Sync failures

**"Network error" / "Connection refused":**
1. Check `WANDB_BASE_URL` points to the correct server
2. Verify API key: `wandb verify`
3. Check firewall/proxy settings
4. For self-hosted: verify the W&B server is healthy: `curl $WANDB_BASE_URL/healthz`

**"Rate limited":**
- W&B cloud has rate limits. Add delays between rapid `log()` calls
- Batch metrics: `run.log({"a": 1, "b": 2})` instead of separate calls
- Reduce `log_freq` in `wandb.watch()`

## Large Artifact Handling

### Upload timeouts

For large artifacts (multi-GB models):

```python
# Increase timeout
import os
os.environ["WANDB_HTTP_TIMEOUT"] = "300"  # seconds

# Use references instead of uploading
artifact = wandb.Artifact("large-model", type="model")
artifact.add_reference("s3://bucket/model-weights/")  # pointer only, no upload
run.log_artifact(artifact)
```

### Artifact cache management

W&B caches downloaded artifacts locally:

```bash
# Default cache location
~/.cache/wandb/artifacts/

# Change cache directory
export WANDB_CACHE_DIR="/data/wandb-cache"

# Clear cache
wandb artifact cache cleanup 10G  # keep only 10GB
```

### Deduplication

W&B deduplicates artifact contents by content hash. If you log the same file twice, it's stored once. This means:
- Re-logging unchanged files is cheap
- Artifact versions that share files don't multiply storage

## Run Resume and Crash Recovery

### Resume a crashed run

```python
# Must use the same run ID
run = wandb.init(
    project="my-project",
    id="crashed-run-id",  # original run ID
    resume="must",        # fail if run doesn't exist
)

# Or allow resume (create new if not found)
run = wandb.init(
    project="my-project",
    id="my-run-id",
    resume="allow",
)
```

**Resume modes:**
| Mode | Behavior |
|------|----------|
| `"must"` | Must resume existing run; error if not found |
| `"allow"` | Resume if exists, create new otherwise |
| `"never"` | Always create new run; error if ID exists |
| `None` | Default — create new run |

### Auto-resume with environment variable

```bash
# Set in your training script launcher
export WANDB_RUN_ID="unique-run-id"
export WANDB_RESUME="allow"
```

### Saving run ID for crash recovery

```python
import wandb
import json

# Save run ID at start
run = wandb.init(project="training")
with open("run_state.json", "w") as f:
    json.dump({"run_id": run.id, "project": run.project}, f)

# On restart, read and resume
with open("run_state.json") as f:
    state = json.load(f)
run = wandb.init(
    project=state["project"],
    id=state["run_id"],
    resume="must",
)
```

## Integration Pitfalls

### HuggingFace Trainer double-logging

If you call `wandb.init()` before `Trainer`, you may get duplicate runs.

**Fix**: Let Trainer handle init, or pass the existing run:
```python
# Option 1: Let Trainer manage it (preferred)
# Don't call wandb.init() — Trainer does it
training_args = TrainingArguments(report_to="wandb", run_name="my-run")

# Option 2: Pass run to Trainer via env
import os
os.environ["WANDB_RUN_ID"] = "my-custom-id"
os.environ["WANDB_RESUME"] = "allow"
```

### Lightning logger not logging

Ensure `log_model` is set and the logger is passed correctly:
```python
wandb_logger = WandbLogger(project="test", log_model="all")
trainer = L.Trainer(logger=wandb_logger)  # not loggers=[wandb_logger]
```

### Ray Train callback not syncing

- Ensure `WANDB_API_KEY` is available on all worker nodes
- For self-hosted: set `WANDB_BASE_URL` in the Ray runtime env
- Check that `wandb` is installed in the worker image

```python
runtime_env = {
    "pip": ["wandb"],
    "env_vars": {
        "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
        "WANDB_BASE_URL": "https://wandb.example.com",
    },
}
```

### wandb.watch() memory issues

`wandb.watch(model, log="all")` logs all gradients and parameters. For large models this is expensive.

**Fix**: Reduce frequency or scope:
```python
run.watch(model, log="gradients", log_freq=500)  # only gradients, less often
# Or skip watch entirely and log specific metrics manually
```

## Common Errors

### "wandb: ERROR api key not configured"

```bash
# Set API key
export WANDB_API_KEY="your-key"
# Or login interactively
wandb login
# Or use a file
echo "your-key" > ~/.netrc  # wandb reads netrc
```

### "CommError: Could not find run"

Usually means the run was deleted or the project/entity is wrong:
```python
# Double-check entity and project
wandb.init(entity="correct-team", project="correct-project")
```

### "ValueError: Artifact name must be valid"

Artifact names must match `[a-zA-Z0-9_.-]+`:
```python
# Wrong
wandb.Artifact("my model (v2)", type="model")  # spaces and parens

# Right
wandb.Artifact("my-model-v2", type="model")
```

### "wandb: WARNING Step must only increase"

You're logging with a step value that's <= a previous step. W&B requires monotonically increasing steps.

**Fix**: Don't pass `step` manually unless you're certain it increases, or use `define_metric()`:
```python
run.define_metric("val/accuracy", step_metric="epoch")
run.log({"epoch": epoch, "val/accuracy": acc})
```

## Performance Tips

### Reduce logging overhead

```python
# Log less frequently
if step % 100 == 0:
    run.log({"loss": loss}, step=step)

# Disable system metrics if not needed
wandb.init(settings=wandb.Settings(
    _disable_stats=True,       # no system metrics
    _disable_meta=True,        # no metadata collection
))
```

### Disable wandb entirely for debugging

```bash
export WANDB_DISABLED=true
# or
wandb.init(mode="disabled")
```

This creates a no-op run — all `log()` calls succeed but nothing is recorded.
