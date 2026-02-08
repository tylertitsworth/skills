# Ray Train Troubleshooting

Common issues when running distributed training with Ray Train.

## Table of Contents

- [NCCL errors](#nccl-errors)
- [OOM errors](#oom-errors)
- [Slow data loading](#slow-data-loading)
- [Checkpoint issues](#checkpoint-issues)
- [Rank mismatches and hangs](#rank-mismatches-and-hangs)
- [Common error patterns](#common-error-patterns)

## NCCL Errors

NCCL is the GPU communication library. Errors usually indicate network or configuration issues.

### "NCCL error: unhandled system error" / "NCCL timeout"

```bash
# Check NCCL connectivity
NCCL_DEBUG=INFO python -c "import torch.distributed; ..."
```

**Common causes:**
1. **Firewall blocking ports** — NCCL uses high ports for GPU-to-GPU communication. Ensure ports are open between nodes.
2. **Network interface mismatch** — Set the correct interface:
   ```bash
   export NCCL_SOCKET_IFNAME=eth0  # or your network interface
   ```
3. **InfiniBand issues** — If not using IB, disable it:
   ```bash
   export NCCL_IB_DISABLE=1
   ```
4. **Timeout too short** — Increase for large models:
   ```bash
   export NCCL_TIMEOUT=1800  # 30 minutes
   ```

### "NCCL error: invalid usage"

Usually means a collective operation was called with mismatched arguments across ranks. Check that all workers execute the same code path.

## OOM Errors

### GPU OOM

1. **Reduce batch size** — The most common fix
2. **Use gradient accumulation** — Simulate larger batches:
   ```python
   for i, batch in enumerate(dataloader):
       loss = model(batch).loss / accumulation_steps
       loss.backward()
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```
3. **Enable mixed precision** (fp16/bf16):
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   with autocast():
       loss = model(batch).loss
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```
4. **Use DeepSpeed ZeRO or FSDP** — See `large-model-training.md`
5. **Enable activation checkpointing** — Trade compute for memory:
   ```python
   from torch.utils.checkpoint import checkpoint
   # or model.gradient_checkpointing_enable() for HuggingFace models
   ```

### CPU OOM

- **Don't pass large objects via `train_loop_config`** — Initialize datasets/models inside `train_func`
- **Use Ray Data streaming** — Avoid loading entire dataset into memory
- **Reduce `num_workers`** — Each worker consumes CPU memory for its copy

## Slow Data Loading

### GPU idle, waiting for data

1. **Use `num_workers` in DataLoader:**
   ```python
   DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)
   ```
2. **Use Ray Data** for streaming from remote storage — avoids loading full dataset
3. **Move preprocessing to CPU workers** — Use Ray Data `map_batches` for CPU-heavy transforms
4. **Pre-tokenize / pre-process** — Save preprocessed data as Parquet, load directly

### Ray Data loading slow

- Increase read parallelism: `ray_remote_args={"num_cpus": 0.25}`
- Use Parquet with column pruning
- See the **ray-data** skill's performance reference

## Checkpoint Issues

### "Storage path must be specified for multi-node training"

Multi-node training requires shared storage. Set:
```python
RunConfig(storage_path="s3://bucket/experiments")
# or NFS: storage_path="/shared/nfs/experiments"
```

### Checkpoints are too large

- For DDP, only save from rank 0 (default recommendation)
- Use `CheckpointConfig(num_to_keep=3)` to limit disk usage
- Save only model weights, not optimizer state, if you don't need fault tolerance

### model.module.state_dict() vs model.state_dict()

After `prepare_model`, the model is wrapped in DDP. Use `model.module.state_dict()` to avoid `module.` prefix in keys. When loading:
```python
# If saved with model.module.state_dict():
model.load_state_dict(state_dict)  # works directly

# If saved with model.state_dict() (has module. prefix):
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
```

## Rank Mismatches and Hangs

### Training hangs with no error

Usually means one rank is waiting for a collective operation that another rank skipped:

1. **Conditional code paths** — Ensure all ranks execute the same collective ops (e.g., `ray.train.report` must be called by ALL workers, even if checkpoint is None)
2. **Deadlocked barriers** — Check for `torch.distributed.barrier()` calls that some ranks skip
3. **Worker crash** — Check Ray logs: `ray logs` or dashboard

### "RuntimeError: Expected all tensors on the same device"

Model and data on different devices. This shouldn't happen if using `prepare_model` and `prepare_data_loader`, but can occur with manual device management.

## Common Error Patterns

| Error | Cause | Fix |
|---|---|---|
| `NCCL timeout` | Network issue between nodes | Check firewall, set `NCCL_SOCKET_IFNAME` |
| `CUDA out of memory` | Batch too large or model too large | Reduce batch, use ZeRO/FSDP, mixed precision |
| `Storage path must be specified` | Multi-node without shared storage | Set `RunConfig(storage_path=...)` |
| `No module named 'ray.train.huggingface'` | Missing install | `pip install "ray[train]" transformers` |
| Training loss NaN | Learning rate too high or numerical instability | Lower LR, use gradient clipping, check fp16 overflow |
| `Address already in use` | Port conflict from previous run | Kill stale processes, or set `MASTER_PORT` |
| Workers die silently | OOM kill by OS | Check `dmesg` for OOM killer, reduce memory |
| `set_epoch` error | Missing sampler call | Call `train_loader.sampler.set_epoch(epoch)` each epoch |
