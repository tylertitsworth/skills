"""Ray Train distributed LLM fine-tuning example.

Demonstrates:
  - TorchTrainer with FSDP strategy
  - Checkpoint management with CheckpointConfig
  - Fault tolerance with FailureConfig
  - Ray Data integration for streaming data

Usage (submit via RayJob or ray job submit):
    python train_llm.py

Requires: ray[train], torch, transformers, datasets
"""

import ray
import ray.train
import ray.train.torch
from ray.train import CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer


def train_func(config: dict):
    """Per-worker training function."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Ray Train sets up distributed env automatically
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare model for distributed training
    model = ray.train.torch.prepare_model(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    # Get Ray Data shard for this worker
    train_ds = ray.train.get_dataset_shard("train")

    for epoch in range(config["epochs"]):
        for batch in train_ds.iter_torch_batches(batch_size=config["batch_size"]):
            input_ids = batch["input_ids"].to(ray.train.torch.get_device())
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Report metrics and save checkpoint
        ray.train.report(
            {"loss": loss.item(), "epoch": epoch},
            checkpoint=ray.train.Checkpoint.from_directory("/tmp/ckpt"),
        )


def main():
    # Load dataset with Ray Data (streaming, no full materialization)
    train_dataset = ray.data.read_parquet("s3://my-bucket/training-data/")

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "model_name": "meta-llama/Llama-3.2-1B",
            "lr": 2e-5,
            "weight_decay": 0.01,
            "batch_size": 4,
            "epochs": 3,
        },
        scaling_config=ScalingConfig(
            num_workers=4,
            use_gpu=True,
            resources_per_worker={"GPU": 1, "CPU": 8},
        ),
        run_config=RunConfig(
            name="llm-finetune",
            checkpoint_config=CheckpointConfig(
                num_to_keep=3,  # keep last 3 checkpoints
                checkpoint_score_attribute="loss",
                checkpoint_score_order="min",
            ),
            failure_config=FailureConfig(
                max_failures=3,  # auto-restart on failure
            ),
        ),
        datasets={"train": train_dataset},
    )

    result = trainer.fit()
    print(f"Training complete. Best checkpoint: {result.best_checkpoints}")


if __name__ == "__main__":
    main()
