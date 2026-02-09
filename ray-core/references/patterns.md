# Ray Core Patterns & Anti-Patterns

## Table of Contents

- [Patterns](#patterns)
  - [Batch parallel processing](#batch-parallel-processing)
  - [Map-reduce](#map-reduce)
  - [Tree reduce](#tree-reduce)
  - [Pipeline parallelism](#pipeline-parallelism)
  - [Parameter server](#parameter-server)
  - [Streaming with ray.wait](#streaming-with-raywait)
  - [Nested parallelism](#nested-parallelism)
- [Anti-patterns](#anti-patterns)
  - [ray.get in a loop](#rayget-in-a-loop)
  - [Unnecessary ray.get](#unnecessary-rayget)
  - [Closure capturing large objects](#closure-capturing-large-objects)
  - [Returning ray.put from tasks](#returning-rayput-from-tasks)
  - [Out-of-band ObjectRef serialization](#out-of-band-objectref-serialization)
  - [Too many small tasks](#too-many-small-tasks)

## Patterns

### Batch Parallel Processing

Submit many tasks, collect all results:

```python
@ray.remote
def process_item(item):
    return transform(item)

# Submit all at once, then collect
futures = [process_item.remote(item) for item in items]
results = ray.get(futures)
```

For large batches, use `ray.wait` to process results as they arrive (see [Streaming](#streaming-with-raywait)).

### Map-Reduce

```python
import ray
from collections import Counter

@ray.remote
def map_fn(chunk):
    counts = Counter()
    for word in chunk.split():
        counts[word] += 1
    return counts

@ray.remote
def reduce_fn(counter1, counter2):
    return counter1 + counter2

# Map phase
chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
mapped = [map_fn.remote(chunk) for chunk in chunks]

# Reduce phase (pairwise)
while len(mapped) > 1:
    mapped = [
        reduce_fn.remote(mapped[i], mapped[i + 1])
        for i in range(0, len(mapped) - 1, 2)
    ] + (mapped[-1:] if len(mapped) % 2 else [])

result = ray.get(mapped[0])
```

### Tree Reduce

More efficient than linear reduce for large fan-ins:

```python
@ray.remote
def tree_reduce(fn, refs, batch_size=4):
    """Reduce ObjectRefs in a tree pattern."""
    while len(refs) > 1:
        new_refs = []
        for i in range(0, len(refs), batch_size):
            batch = refs[i:i + batch_size]
            if len(batch) == 1:
                new_refs.append(batch[0])
            else:
                new_refs.append(fn.remote(*batch))
        refs = new_refs
    return ray.get(refs[0])
```

### Pipeline Parallelism

Chain tasks where stage N+1 starts processing as soon as stage N produces output:

```python
@ray.remote
def stage1(data):
    return preprocess(data)

@ray.remote
def stage2(preprocessed):
    return train(preprocessed)

@ray.remote
def stage3(model):
    return evaluate(model)

# Pipeline — each stage starts when its input is ready (no explicit ray.get)
ref1 = stage1.remote(data)
ref2 = stage2.remote(ref1)    # passes ObjectRef, not value
ref3 = stage3.remote(ref2)
final = ray.get(ref3)
```

### Parameter Server

Central actor holding shared state, updated by distributed workers:

```python
@ray.remote
class ParameterServer:
    def __init__(self, dim):
        self.params = np.zeros(dim)

    def get_params(self):
        return self.params

    def apply_gradients(self, gradients):
        self.params += gradients

@ray.remote(num_gpus=1)
def worker(ps, data):
    for batch in data:
        params = ray.get(ps.get_params.remote())
        gradients = compute_gradients(params, batch)
        ps.apply_gradients.remote(gradients)

ps = ParameterServer.remote(dim=1000)
data_shards = split_data(dataset, num_workers=4)
ray.get([worker.remote(ps, shard) for shard in data_shards])
```

### Streaming with ray.wait

Process results as they complete instead of waiting for all:

```python
futures = [slow_task.remote(i) for i in range(100)]
results = []

while futures:
    ready, futures = ray.wait(futures, num_returns=1)
    result = ray.get(ready[0])
    results.append(result)
    # Process incrementally — useful for progress bars, early stopping, etc.
```

### Nested Parallelism

Tasks that spawn other tasks:

```python
@ray.remote
def process_partition(partition):
    # Each partition spawns subtasks
    subtasks = [process_row.remote(row) for row in partition]
    return ray.get(subtasks)

@ray.remote
def process_row(row):
    return transform(row)

# Top level kicks off partitions
partitions = chunk_data(data, num_partitions=10)
results = ray.get([process_partition.remote(p) for p in partitions])
```

**Warning:** Nested parallelism can cause deadlocks if inner tasks compete for resources with outer tasks. Set `num_cpus=0` on the outer task if it only orchestrates.

## Anti-Patterns

### ray.get in a Loop

**Bad:** Serializes execution — each iteration waits for the previous task:

```python
# ❌ BAD: Sequential execution
results = []
for item in items:
    result = ray.get(process.remote(item))  # blocks each iteration
    results.append(result)
```

**Good:** Submit all tasks first, then collect:

```python
# ✅ GOOD: Parallel execution
futures = [process.remote(item) for item in items]
results = ray.get(futures)
```

### Unnecessary ray.get

**Bad:** Fetching a value just to pass it to another task:

```python
# ❌ BAD: Unnecessary serialization round-trip
result = ray.get(stage1.remote(data))
final = ray.get(stage2.remote(result))
```

**Good:** Pass ObjectRefs directly:

```python
# ✅ GOOD: Pass ref directly, no intermediate fetch
ref = stage1.remote(data)
final = ray.get(stage2.remote(ref))
```

### Closure Capturing Large Objects

**Bad:** Accidentally capturing large objects in the function closure:

```python
large_array = np.zeros((10000, 10000))

# ❌ BAD: large_array is serialized with every task invocation
@ray.remote
def process(i):
    return large_array[i].sum()
```

**Good:** Put the object in the store first:

```python
large_ref = ray.put(np.zeros((10000, 10000)))

# ✅ GOOD: Only the ref (small) is captured
@ray.remote
def process(data, i):
    return data[i].sum()

futures = [process.remote(large_ref, i) for i in range(100)]
```

### Returning ray.put from Tasks

**Bad:** Calling `ray.put()` inside a task and returning the ref:

```python
# ❌ BAD: Disables small-return inlining, breaks fault tolerance
@ray.remote
def bad_task():
    result = compute()
    return ray.put(result)
```

**Good:** Just return the value — Ray handles storage:

```python
# ✅ GOOD: Ray auto-stores the return value
@ray.remote
def good_task():
    return compute()
```

### Out-of-Band ObjectRef Serialization

**Bad:** Serializing ObjectRefs with pickle/JSON and passing outside Ray:

```python
# ❌ BAD: Ray can't track GC for this ref
ref = task.remote()
serialized = pickle.dumps(ref)
```

**Good:** Pass ObjectRefs as arguments to other Ray tasks/actors.

### Too Many Small Tasks

**Bad:** Thousands of sub-millisecond tasks — scheduling overhead dominates:

```python
# ❌ BAD: 1M tiny tasks
futures = [add_one.remote(i) for i in range(1_000_000)]
```

**Good:** Batch work into chunked tasks:

```python
# ✅ GOOD: 1000 tasks processing 1000 items each
@ray.remote
def process_batch(items):
    return [add_one(i) for i in items]

batches = [items[i:i+1000] for i in range(0, len(items), 1000)]
futures = [process_batch.remote(batch) for batch in batches]
```
