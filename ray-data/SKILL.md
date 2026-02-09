---
name: ray-data
description: >
  Build scalable data pipelines with Ray Data for ML workloads. Use when:
  (1) Creating Datasets from files (Parquet, JSON, CSV, images, text, custom sources),
  (2) Building preprocessing pipelines (map, filter, flat_map, map_batches with UDFs),
  (3) GPU-accelerated preprocessing (image transforms, tokenization, batch inference on GPU),
  (4) Streaming data into Ray Train or Ray Serve without materializing full datasets,
  (5) Working with large datasets that don't fit in memory (streaming execution, repartitioning),
  (6) Debugging data pipeline performance (slow stages, memory pressure, block sizing).
---

# Ray Data

Scalable data processing library for ML workloads. Streaming execution engine that efficiently processes data across CPU and GPU.

**Docs:** https://docs.ray.io/en/latest/data/data.html
**Version:** Ray 2.53.0

## Core Concepts

- **Dataset**: Distributed collection of rows stored as Arrow blocks in Ray's object store
- **Block**: Unit of data (Arrow table or pandas DataFrame), typically 1–128 MiB
- **Streaming execution**: Transformations are lazy; data flows through the pipeline without full materialization
- **Operator fusion**: Adjacent map operations are fused into single tasks automatically

## Creating Datasets

```python
import ray

# From files
ds = ray.data.read_parquet("s3://bucket/data/")
ds = ray.data.read_parquet("s3://bucket/data/", columns=["col1", "col2"])  # projection pushdown
ds = ray.data.read_csv("/path/to/files/")
ds = ray.data.read_json("s3://bucket/data.jsonl")
ds = ray.data.read_images("s3://bucket/images/")
ds = ray.data.read_text("s3://bucket/corpus/")

# From memory
ds = ray.data.from_items([{"text": "hello"}, {"text": "world"}])
ds = ray.data.from_pandas(df)
ds = ray.data.from_numpy(np_array)
ds = ray.data.from_torch(torch_dataset)

# Synthetic
ds = ray.data.range(1_000_000)
ds = ray.data.range_tensor(1000, shape=(224, 224, 3))
```

### Read Options

```python
# Parallel reads with lower CPU per task
ds = ray.data.read_parquet(path, ray_remote_args={"num_cpus": 0.25})

# Control block count
ds = ray.data.read_parquet(path, override_num_blocks=64)
```

## Transformations

All transformations are **lazy** — they build a logical plan executed on consumption.

### Row-Level Operations

```python
# Map each row (Dict[str, Any])
ds = ds.map(lambda row: {"text": row["text"].lower()})

# Filter rows
ds = ds.filter(lambda row: row["score"] > 0.5)

# One-to-many (explode)
ds = ds.flat_map(lambda row: [{"word": w} for w in row["text"].split()])

# Column operations
ds = ds.select_columns(["text", "label"])
ds = ds.drop_columns(["temp_col"])
ds = ds.rename_columns({"old_name": "new_name"})
```

### Batch Operations (Preferred for Performance)

`map_batches` is the workhorse — vectorized operations on batches:

```python
# Lambda on batches (Dict[str, np.ndarray] by default)
ds = ds.map_batches(lambda batch: {"id": batch["id"] * 2})

# With pandas format
ds = ds.map_batches(
    lambda df: df.assign(normalized=df["value"] / df["value"].max()),
    batch_format="pandas",
    batch_size=1024,
)
```

### Class-Based UDFs (Stateful Transforms)

For transforms that need expensive initialization (model loading, tokenizer setup):

```python
class Tokenizer:
    def __init__(self):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def __call__(self, batch):
        tokens = self.tokenizer(
            list(batch["text"]),
            padding=True,
            truncation=True,
            return_tensors="np",
        )
        return {**batch, **tokens}

ds = ds.map_batches(
    Tokenizer,
    batch_size=64,
    compute=ray.data.ActorPoolStrategy(size=4),  # 4 actor workers
)
```

### GPU-Accelerated Preprocessing

```python
class GPUImageProcessor:
    def __init__(self):
        import torch
        self.device = torch.device("cuda")
        self.transform = build_transform()

    def __call__(self, batch):
        import torch
        images = torch.tensor(batch["image"]).to(self.device)
        processed = self.transform(images)
        return {"image": processed.cpu().numpy()}

ds = ds.map_batches(
    GPUImageProcessor,
    batch_size=32,
    num_gpus=1,                                    # 1 GPU per worker
    compute=ray.data.ActorPoolStrategy(size=2),    # 2 GPU workers
    batch_format="numpy",
)
```

### Batch Inference

```python
class LLMPredictor:
    def __init__(self):
        from transformers import pipeline
        self.pipe = pipeline("text-classification", device="cuda:0")

    def __call__(self, batch):
        results = self.pipe(list(batch["text"]))
        batch["label"] = [r["label"] for r in results]
        batch["score"] = [r["score"] for r in results]
        return batch

ds = ray.data.read_parquet("s3://bucket/input/")
ds = ds.map_batches(
    LLMPredictor,
    batch_size=64,
    num_gpus=1,
    compute=ray.data.ActorPoolStrategy(min_size=1, max_size=4),
    batch_format="numpy",
)
ds.write_parquet("s3://bucket/output/")
```

## Consuming Data

### Iterating

```python
# Iterate batches (for custom training loops)
for batch in ds.iter_batches(batch_size=256, batch_format="numpy"):
    train_step(batch)

# PyTorch batches
for batch in ds.iter_torch_batches(batch_size=256):
    model(batch["features"].to(device))

# Convert to TensorFlow dataset
tf_ds = ds.to_tf(feature_columns="features", label_columns="label")

# Peek at data
ds.show(5)
ds.schema()
ds.count()
```

### Feeding into Ray Train

```python
import ray.train
from ray.train.torch import TorchTrainer

def train_fn(config):
    ds_shard = ray.train.get_dataset_shard("train")
    for epoch in range(config["epochs"]):
        for batch in ds_shard.iter_torch_batches(batch_size=64):
            # Training loop
            pass

ds = ray.data.read_parquet("s3://bucket/training-data/")
trainer = TorchTrainer(
    train_fn,
    datasets={"train": ds},          # streaming, not materialized
    scaling_config=ray.train.ScalingConfig(num_workers=4, use_gpu=True),
)
```

## Aggregations

```python
# Global aggregations
ds.count()
ds.sum("value")
ds.min("value")
ds.max("value")
ds.mean("value")
ds.std("value")
ds.unique("category")
ds.summary()             # statistical summary by data type

# GroupBy aggregations
ds.groupby("category").count()
ds.groupby("category").mean("value")
ds.groupby("category").sum("value")
ds.groupby(["cat1", "cat2"]).max("value")
```

## Joins

```python
ds1 = ray.data.from_items([{"id": 1, "name": "a"}, {"id": 2, "name": "b"}])
ds2 = ray.data.from_items([{"id": 1, "score": 0.9}, {"id": 2, "score": 0.8}])

# Inner join (default)
joined = ds1.join(ds2, on="id")

# Left join
joined = ds1.join(ds2, on="id", join_type="left")
```

## Union and Zip

```python
# Concatenate datasets
combined = ds1.union(ds2)

# Zip datasets column-wise (must have same row count)
zipped = ds1.zip(ds2)
```

## Custom Datasources

```python
from ray.data.datasource import Datasource

class MyDatasource(Datasource):
    def get_read_tasks(self, parallelism):
        # Return list of ReadTask
        ...

ds = ray.data.read_datasource(MyDatasource(), parallelism=10)
```

## Data Context Configuration

```python
ctx = ray.data.DataContext.get_current()
ctx.target_max_block_size = 128 * 1024 * 1024   # 128MB blocks
ctx.execution_options.resource_limits.object_store_memory = 10e9
ctx.execution_options.verbose_progress = True
```

## Shuffling and Repartitioning

```python
# Random shuffle (all-to-all, expensive)
ds = ds.random_shuffle()

# Local shuffle (within each block, cheap)
for batch in ds.iter_batches(batch_size=256, local_shuffle_buffer_size=10000):
    ...

# Repartition (change parallelism)
ds = ds.repartition(100)

# Sort
ds = ds.sort("timestamp")
```

**Tip:** For training, prefer `local_shuffle_buffer_size` over `random_shuffle()`. Full shuffle is expensive and often unnecessary.

## Writing Output

```python
ds.write_parquet("s3://bucket/output/")
ds.write_csv("/local/output/")
ds.write_json("s3://bucket/output/")
ds.write_images("/local/images/", column="image")
```

## Performance and Debugging

For performance tuning, memory management, and troubleshooting, see `references/performance.md`.
