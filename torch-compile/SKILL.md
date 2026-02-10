---
name: torch-compile
description: torch.compile and TorchInductor — compilation modes, backends, graph breaks, inductor options, and pitfalls. Use when configuring torch.compile for training or inference optimization, debugging graph breaks, or tuning inductor settings.
---

# torch.compile & TorchInductor

## How It Works

`torch.compile` has three stages:
1. **TorchDynamo** (graph capture): Intercepts Python bytecode, traces PyTorch operations into FX graphs
2. **AOTAutograd**: Captures backward pass ahead-of-time, producing forward + backward graphs
3. **TorchInductor** (default backend): Compiles FX graphs to Triton kernels (GPU) or C++/OpenMP (CPU)

## API

```python
compiled_model = torch.compile(
    model,
    fullgraph=False,    # Require single graph (error on graph breaks)
    dynamic=None,       # Dynamic shapes: True/False/None (auto-detect)
    backend="inductor", # Compilation backend
    mode=None,          # Optimization preset
    options=None,       # Backend-specific options dict
    disable=False,      # No-op (for A/B testing)
)
```

### Parameters

| Parameter | Type | Default | Effect |
|-----------|------|---------|--------|
| `fullgraph` | bool | False | Error on graph breaks instead of falling back to eager |
| `dynamic` | bool/None | None | `True` = maximize dynamic shapes; `False` = always specialize; `None` = auto-detect |
| `backend` | str | `"inductor"` | Compilation backend |
| `mode` | str | None | `"default"`, `"reduce-overhead"`, `"max-autotune"`, `"max-autotune-no-cudagraphs"` |
| `options` | dict | None | Backend-specific configuration |
| `disable` | bool | False | Turn compile into no-op |

### Modes

| Mode | CUDA Graphs | Triton Autotune | Overhead | Best For |
|------|-------------|----------------|----------|----------|
| `"default"` | No | No | Low compile time | General use |
| `"reduce-overhead"` | Yes | No | Lower Python overhead | Small batches, latency-sensitive |
| `"max-autotune"` | Yes | Yes | Highest compile time | Maximum throughput |
| `"max-autotune-no-cudagraphs"` | No | Yes | Moderate | When CUDA graphs cause issues |

**`reduce-overhead`** wraps compiled regions in CUDA graphs, eliminating Python dispatch overhead. Tradeoff: cached workspace memory increases GPU memory usage.

**`max-autotune`** profiles multiple Triton kernel configs and matmul implementations (Triton vs cuBLAS) to select the fastest. First compilation is much slower.

### Backends

List available backends:

```python
torch._dynamo.list_backends()       # Stable backends
torch._dynamo.list_backends(None)   # All (including experimental)
```

| Backend | Output | Use Case |
|---------|--------|----------|
| `"inductor"` | Triton/C++ kernels | Default, best balance |
| `"cudagraphs"` | CUDA graph capture | Reduce launch overhead only |
| `"aot_eager"` | Eager PyTorch | Debug AOTAutograd |
| `"eager"` | No compilation | Baseline comparison |

## Inductor Options

Pass via `options={}` dict or `torch._inductor.config`:

```python
model = torch.compile(model, options={
    "max_autotune": True,
    "epilogue_fusion": True,
    "shape_padding": True,
    "triton.cudagraphs": True,
    "trace.enabled": True,
    "trace.graph_diagram": True,
})
```

### Key Inductor Options

| Option | Type | Default | Effect |
|--------|------|---------|--------|
| `max_autotune` | bool | False | Profile multiple kernel configs for matmuls |
| `epilogue_fusion` | bool | depends | Fuse pointwise ops into matmul templates |
| `shape_padding` | bool | False | Pad tensor dims for better GPU alignment |
| `triton.cudagraphs` | bool | depends on mode | Wrap in CUDA graphs |
| `trace.enabled` | bool | False | Generate debug trace output |
| `trace.graph_diagram` | bool | False | Visualize fusion graph |
| `fallback_random` | bool | False | Use Python random (reproducibility debugging) |
| `coordinate_descent_tuning` | bool | False | Extra tuning after autotune |
| `force_disable_caches` | bool | False | Disable kernel caching |
| `triton.unique_kernel_names` | bool | False | Unique names for profiling |

Full list: `torch._inductor.list_options()`

Mode presets: `torch._inductor.list_mode_options("reduce-overhead")`

## Graph Breaks

A **graph break** occurs when TorchDynamo encounters code it can't trace into the FX graph. The model runs as multiple compiled subgraphs with eager Python in between.

### Common Causes

| Cause | Example | Fix |
|-------|---------|-----|
| Data-dependent control flow | `if x.sum() > 0:` | Use `torch.where()` or `torch.cond()` |
| Unsupported Python built-in | `print(tensor)` | Remove or guard with `if not torch.compiler.is_compiling():` |
| Non-tensor operations | `tensor.tolist()` | Avoid materializing to Python |
| Dynamic module creation | `nn.Linear(x.shape[0], 10)` | Pre-create modules |
| Custom autograd function | Without `@torch.library.custom_op` | Register as custom op |
| Third-party C extensions | Non-PyTorch CUDA kernels | Use `torch.compiler.allow_in_graph()` |
| `torch.no_grad()` in forward | Context manager switch | Use `@torch.inference_mode` at outer level |
| `**kwargs` in some patterns | Dynamic dict unpacking | Use explicit args |

### Debugging Graph Breaks

```python
# Log all graph breaks with explanations
import torch._dynamo
torch._dynamo.config.verbose = True

# Or via environment variable
# TORCH_LOGS=graph_breaks python train.py

# Strict mode — error on any break
model = torch.compile(model, fullgraph=True)
```

```bash
# Environment variable logging
TORCH_LOGS="+dynamo,graph_breaks,guards" python train.py
```

### Recompilation

Guard failures trigger recompilation. After `recompile_limit` (default: 8) recompiles for a frame, it falls back to eager permanently.

Common recompilation triggers:
- Changing input shapes (use `dynamic=True`)
- Changing dtypes
- Changing device

```python
# Increase limit if needed (rare)
torch._dynamo.config.recompile_limit = 16

# Reset dynamo state between runs
torch._dynamo.reset()
```

## Practical Patterns

### Training

```python
model = torch.compile(model, mode="max-autotune", fullgraph=True)

# Compile optimizer step separately (requires graph break)
optimizer.step = torch.compile(optimizer.step)
```

Note: Optimizer compilation currently requires graph breaks, so apply it **without** `fullgraph=True` on the optimizer.

### Inference

```python
model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
model.eval()

# Warmup — first call triggers compilation
with torch.no_grad():
    _ = model(sample_input)

# Subsequent calls use compiled code
output = model(real_input)
```

### Selective Compilation

Compile only specific submodules to avoid graph breaks in outer logic:

```python
# Don't compile the whole model
model.encoder = torch.compile(model.encoder, fullgraph=True)
model.decoder = torch.compile(model.decoder, fullgraph=True)
```

### torch.compile + FSDP

```python
# Compile FIRST, then wrap with FSDP
model = torch.compile(model)
model = FSDP(model, ...)

# Or compile individual modules before FSDP wrapping
for layer in model.layers:
    layer = torch.compile(layer)
```

### Disabling for Specific Functions

```python
@torch.compiler.disable
def non_compilable_function(x):
    # Complex Python logic that causes graph breaks
    ...
```

## Environment Variables

| Variable | Effect |
|----------|--------|
| `TORCH_LOGS="+dynamo"` | Dynamo debug logging |
| `TORCH_LOGS=graph_breaks` | Log graph breaks only |
| `TORCH_LOGS=guards` | Log guard failures |
| `TORCH_LOGS=recompiles` | Log recompilations |
| `TORCH_LOGS=perf_hints` | Log performance recommendations |
| `TORCH_LOGS=output_code` | Dump generated Triton/C++ code |
| `TORCHINDUCTOR_CACHE_DIR` | Kernel cache directory |
| `TORCHINDUCTOR_FX_GRAPH_CACHE` | Enable FX graph cache (1/0) |

For K8s containers, set these as env vars in the pod spec:

```yaml
env:
  - name: TORCHINDUCTOR_CACHE_DIR
    value: /tmp/torchinductor_cache
  - name: TORCH_LOGS
    value: "perf_hints"
```

## Pitfalls

### CUDA Graphs + Input Mutation

CUDA graphs replay fixed operations. If your model mutates inputs or has side effects, `reduce-overhead` mode will produce incorrect results silently.

### Compilation Time

First-call compilation can take minutes for large models with `max-autotune`. In K8s:
- Use **warm-up** in readiness probes (long `initialDelaySeconds`)
- Consider **persistent kernel cache** via PVC mounted at `TORCHINDUCTOR_CACHE_DIR`
- Pre-compile in init container if model is static

### Dynamic Shapes

Default behavior (`dynamic=None`) specializes on first input shape, then recompiles on shape change. For variable-length sequences:

```python
model = torch.compile(model, dynamic=True)
```

This generates kernels with symbolic shapes, avoiding recompilation but potentially slower per-kernel.

### Numerical Differences

Compiled code may produce slightly different floating-point results due to kernel fusion and reordering. For strict reproducibility:

```python
model = torch.compile(model, options={"fallback_random": True})
torch.use_deterministic_algorithms(True)
```

## Cross-References

- [pytorch](../pytorch/) — PyTorch fundamentals and training loops
- [fsdp](../fsdp/) — FSDP + torch.compile integration ordering
- [flash-attention](../flash-attention/) — Attention backends that torch.compile can fuse
- [vllm](../vllm/) — vLLM uses torch.compile internally for inference optimization

## Reference

- [torch.compile tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [TorchDynamo deep dive](https://pytorch.org/docs/stable/torch.compiler_deepdive.html)
- [TorchInductor docs](https://pytorch.org/docs/stable/torch.compiler_inductor_profiling.html)
- [Troubleshooting guide](https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html)
- `scripts/diagnose_graph_breaks.py` — diagnose graph breaks and benchmark compiled vs eager performance
