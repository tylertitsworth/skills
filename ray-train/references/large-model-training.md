# Large Model Training with DeepSpeed & FSDP

Training models too large for a single GPU using model parallelism strategies.

## Table of Contents

- [DeepSpeed ZeRO](#deepspeed-zero)
- [PyTorch FSDP](#pytorch-fsdp)
- [Distributed checkpointing](#distributed-checkpointing)
- [Choosing between DeepSpeed and FSDP](#choosing-between-deepspeed-and-fsdp)

## DeepSpeed ZeRO

Use `deepspeed.initialize` inside the training function — Ray Train handles distributed setup:

```python
import deepspeed
from deepspeed.accelerator import get_accelerator
import ray.train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

def train_func(config):
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

    deepspeed_config = {
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": 2e-5, "weight_decay": 0.01},
        },
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,                          # ZeRO Stage 3
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
        },
        "gradient_accumulation_steps": 4,
        "train_micro_batch_size_per_gpu": 2,
    }

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=deepspeed_config,
    )
    device = get_accelerator().device_name(model.local_rank)

    for epoch in range(config["epochs"]):
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            model.backward(outputs.loss)
            model.step()

        # Distributed checkpoint — each worker saves its shard
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_checkpoint(tmpdir)
            torch.distributed.barrier()
            ray.train.report(
                metrics={"loss": loss},
                checkpoint=ray.train.Checkpoint.from_directory(tmpdir),
            )

trainer = TorchTrainer(
    train_func,
    train_loop_config={"epochs": 3},
    scaling_config=ScalingConfig(num_workers=8, use_gpu=True),
    run_config=ray.train.RunConfig(storage_path="s3://bucket/checkpoints"),
)
result = trainer.fit()
```

### ZeRO Stages

| Stage | Partitions | Memory Savings | Communication |
|---|---|---|---|
| **ZeRO-1** | Optimizer states | ~4× | Low overhead |
| **ZeRO-2** | + Gradients | ~8× | Moderate overhead |
| **ZeRO-3** | + Parameters | ~N× (N=workers) | Higher overhead |

### CPU Offloading

For extremely large models, offload optimizer states or parameters to CPU:

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu", "pin_memory": true},
        "offload_param": {"device": "cpu", "pin_memory": true}
    }
}
```

Trade-off: Saves GPU memory but increases training time due to CPU-GPU data transfer.

## PyTorch FSDP

Use FSDP via `prepare_model` or directly with PyTorch's FSDP API:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
import ray.train.torch

def train_func(config):
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

    # Option 1: Use prepare_model with FSDP (simpler)
    model = ray.train.torch.prepare_model(
        model,
        parallel_strategy="fsdp",
        parallel_strategy_kwargs={
            "sharding_strategy": ShardingStrategy.FULL_SHARD,
        },
    )

    # Option 2: Manual FSDP wrapping (more control)
    # model = FSDP(
    #     model,
    #     sharding_strategy=ShardingStrategy.FULL_SHARD,
    #     auto_wrap_policy=size_based_auto_wrap_policy,
    #     device_id=torch.cuda.current_device(),
    # )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(config["epochs"]):
        for batch in dataloader:
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

### FSDP Sharding Strategies

| Strategy | Behavior |
|---|---|
| `FULL_SHARD` | Shard params, gradients, optimizer states across all GPUs (most memory efficient) |
| `SHARD_GRAD_OP` | Shard gradients and optimizer states only |
| `NO_SHARD` | Equivalent to DDP (no sharding) |
| `HYBRID_SHARD` | Full shard within node, replicate across nodes |

## Distributed Checkpointing

With ZeRO-3 or FSDP, each worker holds only a shard. All workers must save their shard:

```python
# Every worker saves (not just rank 0)
with tempfile.TemporaryDirectory() as tmpdir:
    # DeepSpeed
    model.save_checkpoint(tmpdir)

    # OR FSDP
    # full_state = FSDP.full_state_dict(model)
    # if rank == 0: torch.save(full_state, path)

    torch.distributed.barrier()
    ray.train.report(
        metrics=metrics,
        checkpoint=ray.train.Checkpoint.from_directory(tmpdir),
    )
```

Ray Train uploads shards from all workers in parallel to persistent storage.

## Choosing Between DeepSpeed and FSDP

| Factor | DeepSpeed | FSDP |
|---|---|---|
| **Ease of setup** | Config-driven (JSON) | Code-driven (PyTorch native) |
| **CPU offloading** | Built-in, well-tested | Experimental |
| **Ecosystem** | Mature, widely used for LLMs | Part of PyTorch core |
| **Mixed precision** | Built-in fp16/bf16 | Uses PyTorch AMP |
| **Activation checkpointing** | Built-in | Via PyTorch utils |
| **HuggingFace integration** | Native support | Native support |
| **Best for** | Large LLM training, CPU offload | PyTorch-native workflows |

For LLM fine-tuning, both work well. DeepSpeed is more battle-tested for very large models (70B+). FSDP is simpler if you prefer staying within the PyTorch ecosystem.
