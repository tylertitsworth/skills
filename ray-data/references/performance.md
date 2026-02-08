# Ray Data Performance & Troubleshooting

## Table of Contents

- [Key performance rules](#key-performance-rules)
- [Memory management](#memory-management)
- [Block sizing](#block-sizing)
- [Optimizing transforms](#optimizing-transforms)
- [Optimizing reads](#optimizing-reads)
- [GPU pipeline optimization](#gpu-pipeline-optimization)
- [Debugging slow pipelines](#debugging-slow-pipelines)
- [Common issues](#common-issues)

## Key Performance Rules

1. **Use `map_batches` over `map`** — vectorized batch operations are much faster
2. **Use class-based UDFs with `ActorPoolStrategy`** — amortize initialization costs (model loading)
3. **Use Parquet column pruning** — specify `columns=` in `read_parquet()` to avoid loading unused data
4. **Avoid full shuffles** — use `local_shuffle_buffer_size` instead of `random_shuffle()` when possible
5. **Don't materialize unnecessarily** — keep the pipeline lazy; avoid `ds.materialize()` unless needed
6. **Tune batch_size** — larger batches improve GPU utilization but use more memory

## Memory Management

Ray Data bounds heap memory to approximately `num_execution_slots * max_block_size`.

### Streaming Execution

Transformations execute in streaming fashion by default — data flows through operators without materializing the full dataset:

```
Read → Map → Map_Batches → Write
  ↓       ↓         ↓          ↓
[block] [block]  [block]    [file]
```

Backpressure is automatic. If a downstream operator is slow, upstream operators pause.

### Out-of-Memory Prevention

```python
import ray

ctx = ray.data.DataContext.get_current()

# Block size bounds (defaults are usually fine)
ctx.target_min_block_size = 1 * 1024 * 1024       # 1 MiB
ctx.target_max_block_size = 128 * 1024 * 1024      # 128 MiB

# Shuffle block size (larger to avoid tiny blocks)
ctx.target_shuffle_max_block_size = 1024 * 1024 * 1024  # 1 GiB
```

**If you're hitting OOM:**
- Reduce `batch_size` in `map_batches`
- Ensure no single row is enormous (e.g., very large images)
- Use `ActorPoolStrategy(size=N)` to limit concurrent workers
- Avoid accumulating results with `take_all()` on large datasets

### Object Spilling

When the object store fills up, Ray spills objects to disk. This is automatic but slow. To minimize spilling:
- Keep block sizes within the default range
- Don't `materialize()` large intermediate datasets
- Use streaming consumption (`iter_batches`) instead of `to_pandas()` for large data

## Block Sizing

| Parameter | Default | Purpose |
|---|---|---|
| `target_min_block_size` | 1 MiB | Avoid overhead from too many tiny blocks |
| `target_max_block_size` | 128 MiB | Prevent OOM from too-large blocks |
| `override_num_blocks` | auto | Explicit control over read parallelism |

Blocks are split dynamically if they exceed 1.5× the target max size.

**Rule of thumb:** Don't manually tune block sizes unless you've identified a specific bottleneck.

## Optimizing Transforms

### Operator Fusion

Adjacent map operations are automatically fused into a single task:

```python
# These three operations execute as ONE task per block:
ds = ray.data.read_parquet(path)
ds = ds.map_batches(tokenize)
ds = ds.map_batches(normalize)
```

Fusion is broken by:
- All-to-all operations (shuffle, sort, repartition)
- Changing compute strategy (CPU → GPU)
- Mismatched `override_num_blocks` settings

### ActorPoolStrategy

Use for stateful or expensive initialization:

```python
# Fixed pool
compute=ray.data.ActorPoolStrategy(size=4)

# Autoscaling pool
compute=ray.data.ActorPoolStrategy(min_size=1, max_size=8)
```

Without `ActorPoolStrategy`, `map_batches` uses stateless tasks — the UDF class is re-instantiated every time (bad for model loading).

### Batch Size Tuning

```python
# Small batch_size = more overhead, less memory
# Large batch_size = less overhead, more memory, better GPU utilization
ds.map_batches(fn, batch_size=64)   # good for GPU inference
ds.map_batches(fn, batch_size=4096) # good for simple numpy ops
```

## Optimizing Reads

### Parquet Best Practices

```python
# Column pruning (projection pushdown) — only read what you need
ds = ray.data.read_parquet(path, columns=["text", "label"])

# Row filtering (predicate pushdown)
ds = ray.data.read_parquet(path, filter=pyarrow.compute.field("year") >= 2024)

# More IO parallelism for network-bound reads
ds = ray.data.read_parquet(path, ray_remote_args={"num_cpus": 0.25})
```

### Reading Images

```python
# Read images as numpy arrays
ds = ray.data.read_images("s3://bucket/images/", mode="RGB", size=(224, 224))

# With include_paths for metadata
ds = ray.data.read_images(path, include_paths=True)
# Schema: {image: numpy.ndarray, path: string}
```

## GPU Pipeline Optimization

The key pattern for ML data pipelines: **CPU preprocessing → GPU model inference**.

```python
# 1. Read on CPU (many cheap workers)
ds = ray.data.read_parquet(path, ray_remote_args={"num_cpus": 0.25})

# 2. CPU preprocessing (fused with read if possible)
ds = ds.map_batches(preprocess_fn, batch_size=256)

# 3. GPU inference (few expensive workers)
ds = ds.map_batches(
    GPUModel,
    num_gpus=1,
    batch_size=32,
    compute=ray.data.ActorPoolStrategy(size=4),
)

# 4. Write output
ds.write_parquet("s3://bucket/results/")
```

**Keep GPUs busy:**
- CPU stages should produce data faster than GPU stages consume it
- If GPUs are idle, increase read parallelism or CPU worker count
- If GPUs are OOM, reduce `batch_size`
- Monitor with `ds.materialize().stats()` to see per-operator timing

## Debugging Slow Pipelines

### Execution Stats

```python
# Materialize and print detailed stats
materialized = ds.materialize()
print(materialized.stats())
```

Output shows per-operator: tasks executed, blocks produced, time, rows/s.

### Dashboard

The Ray Dashboard (port 8265) shows:
- Active data operators
- Block-level progress
- Object store memory usage

### Common Bottleneck Patterns

| Symptom | Likely Cause | Fix |
|---|---|---|
| GPU idle, CPU busy | CPU preprocessing too slow | Add more CPU workers, optimize UDF |
| CPU idle, GPU busy | Not enough read parallelism | Lower `num_cpus` for reads, increase `override_num_blocks` |
| OOM errors | Blocks too large or too many in memory | Reduce `batch_size`, avoid `materialize()` |
| Slow start | Many small files | Batch files per read task with `override_num_blocks` |
| Pipeline stalls | Backpressure from slow operator | Profile with `stats()`, optimize bottleneck stage |
| Excessive spilling | Object store full | Reduce parallelism, check for leaked references |

## Common Issues

### "Data is lost" / ObjectLostError

Usually means an object was garbage collected or a node died. Solutions:
- Enable object reconstruction (default in streaming execution)
- Increase object store memory with `RAY_object_store_memory`

### Slow Parquet Reads

- Use `columns=` for projection pushdown
- Ensure Parquet files are ~128 MiB each (too small = overhead, too large = poor parallelism)
- Use `ray_remote_args={"num_cpus": 0.25}` for more IO parallelism

### Incompatible Batch Formats

```python
# Specify the format you want
ds.map_batches(fn, batch_format="pandas")   # pandas DataFrame
ds.map_batches(fn, batch_format="numpy")    # Dict[str, np.ndarray]
ds.iter_torch_batches(batch_size=64)        # Dict[str, torch.Tensor]
```
