"""Ray Data preprocessing pipeline for ML training data.

Demonstrates:
  - Reading from multiple sources (parquet, JSON)
  - GPU-accelerated transforms
  - Streaming to avoid OOM on large datasets
  - Writing preprocessed output

Usage:
    python preprocess_pipeline.py
    ray job submit -- python preprocess_pipeline.py
"""

import ray


def tokenize_batch(batch: dict, tokenizer_name: str = "meta-llama/Llama-3.2-1B") -> dict:
    """Tokenize a batch of text using HuggingFace tokenizers."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=2048,
        return_tensors="np",
    )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


class GPUEmbedder:
    """Compute embeddings on GPU using a sentence transformer."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def __call__(self, batch: dict) -> dict:
        import torch

        encoded = self.tokenizer(
            batch["text"],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)
            embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()

        batch["embedding"] = embeddings
        return batch


def main():
    ray.init()

    # Read from parquet (streaming — doesn't load entire dataset)
    ds = ray.data.read_parquet("s3://my-bucket/raw-data/")

    # CPU preprocessing: tokenization
    ds = ds.map_batches(
        tokenize_batch,
        batch_size=256,
        fn_kwargs={"tokenizer_name": "meta-llama/Llama-3.2-1B"},
        num_cpus=2,
    )

    # GPU preprocessing: embeddings (actor pool for model reuse)
    ds = ds.map_batches(
        GPUEmbedder,
        batch_size=64,
        num_gpus=1,
        compute=ray.data.ActorPoolStrategy(size=2),
        concurrency=2,
    )

    # Write preprocessed data
    ds.write_parquet("s3://my-bucket/preprocessed-data/")

    print(f"Pipeline complete. Schema: {ds.schema()}")
    print(f"Row count: {ds.count()}")


if __name__ == "__main__":
    main()
