---
name: model-formats
description: ML model serialization formats — SafeTensors, GGUF, ONNX, pickle. Use when choosing formats, converting between them, understanding security implications, or configuring format-specific loading in vLLM/Transformers/Triton/Ollama.
---

# Model Formats

## Format Comparison

| Format | Stores | Security | Memory Mapping | Framework | Primary Use |
|--------|--------|----------|---------------|-----------|-------------|
| SafeTensors | Weights only | Safe (no code exec) | Yes (zero-copy) | HF Transformers, vLLM | Training checkpoints, HF Hub distribution |
| GGUF | Weights + metadata + tokenizer | Safe (no code exec) | Yes | llama.cpp, Ollama | Quantized local/edge inference |
| ONNX | Graph + weights | Safe (no code exec) | Partial | ONNX Runtime, Triton | Cross-framework deployment |
| Pickle (`.bin`, `.pt`, `.ckpt`) | Arbitrary Python objects | **Unsafe** (arbitrary code exec) | No | PyTorch native | Legacy, internal-only |
| TensorRT (`.engine`, `.plan`) | Compiled engine | Safe | No | TensorRT, Triton | NVIDIA-optimized production inference |

## SafeTensors

The standard format for Hugging Face ecosystem. Weights-only, memory-mappable, with built-in tensor metadata validation.

### Structure

A SafeTensors file contains:
1. **Header** (JSON): Tensor names, dtypes, shapes, byte offsets
2. **Data**: Raw tensor bytes in declared order

No executable code — the header is validated before any data is read, preventing arbitrary code execution attacks.

### Loading

```python
from safetensors.torch import load_file, save_file

# Load — returns dict of {name: tensor}
tensors = load_file("model.safetensors", device="cuda:0")

# Save
save_file(tensors, "model.safetensors")

# Memory-mapped loading (zero-copy on CPU)
from safetensors import safe_open
with safe_open("model.safetensors", framework="pt", device="cpu") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)  # Loaded on demand
```

### Sharded Models

Large models split across multiple files use an index file:

```
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
model.safetensors.index.json          # Maps tensor names → files
```

The index JSON maps each tensor name to its file and byte offset. HF Transformers, vLLM, and other loaders use this to load only the needed shards.

### vLLM Loading

vLLM loads SafeTensors natively. Control with:

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    load_format="safetensors",        # Default — explicit for clarity
    # load_format="auto",             # Auto-detect format
)
```

`load_format` options: `"auto"`, `"safetensors"`, `"pt"` (pickle), `"npcache"`, `"tensorizer"`, `"bitsandbytes"`.

### RunAI Model Streamer

vLLM supports streaming SafeTensors directly from object storage:

```python
llm = LLM(
    model="s3://bucket/model/",
    load_format="runai_streamer",
)
```

## GGUF (GPT-Generated Unified Format)

Self-contained format for quantized inference. Single file includes weights, tokenizer, model architecture metadata, and quantization parameters.

### Quantization Types

| Type | Bits/Weight | Quality | Speed | Use Case |
|------|------------|---------|-------|----------|
| F16 | 16 | Baseline | Slowest | Reference, conversion source |
| Q8_0 | 8 | Near-lossless | Fast | When VRAM allows |
| Q6_K | 6.6 | Excellent | Fast | Best quality-size balance |
| Q5_K_M | 5.5 | Very good | Faster | Good default for most models |
| Q4_K_M | 4.8 | Good | Faster | Popular sweet spot |
| Q4_0 | 4.0 | Acceptable | Fastest | Maximum compression |
| Q3_K_M | 3.4 | Degraded | Fastest | Edge/mobile |
| Q2_K | 2.6 | Poor | Fastest | Extreme compression only |
| IQ4_XS | 4.25 | Good (importance) | Fast | Importance matrix quantized |

The `_K` suffix means k-quant (mixed precision per tensor group). `_M` = medium quality, `_S` = small, `_L` = large. `IQ` = importance-matrix quantized (requires calibration data).

### Conversion from SafeTensors

```bash
# Using llama.cpp convert script
python convert_hf_to_gguf.py ./model-dir --outtype f16 --outfile model-f16.gguf

# Then quantize
llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M

# With importance matrix (better quality)
llama-quantize --imatrix imatrix.dat model-f16.gguf model-IQ4_XS.gguf IQ4_XS
```

### Metadata

GGUF files carry architecture metadata (context length, vocab size, layer count, RoPE parameters) as key-value pairs. Inspect with:

```bash
# llama.cpp
llama-gguf-metadata model.gguf

# Python
from gguf import GGUFReader
reader = GGUFReader("model.gguf")
for field in reader.fields.values():
    print(f"{field.name}: {field.parts[field.data[0]]}")
```

### Ollama Integration

Ollama can load GGUF files directly or pull quantized models from Hugging Face:

```dockerfile
# Modelfile — from local GGUF
FROM ./model-Q4_K_M.gguf

# Modelfile — from HF Hub
FROM hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M
```

## ONNX (Open Neural Network Exchange)

Graph-level interchange format. Unlike SafeTensors/GGUF (weights only), ONNX encodes the entire computation graph + weights, enabling framework-agnostic deployment.

### Export from PyTorch

```python
import torch
import torch.onnx

# Standard export
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=17,              # Use latest stable opset
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={                 # Dynamic batch & sequence
        "input_ids": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "logits": {0: "batch", 1: "seq_len"},
    },
)
```

### Optimum Export (Recommended for Transformers)

```python
from optimum.onnxruntime import ORTModelForCausalLM

model = ORTModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    export=True,                   # Triggers ONNX export
    provider="CUDAExecutionProvider",
)
model.save_pretrained("./onnx-model/")
```

### Triton Inference Server

Triton supports ONNX models via the ONNX Runtime backend:

```
model_repository/
└── my_model/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

```protobuf
# config.pbtxt
name: "my_model"
platform: "onnxruntime_onnx"
max_batch_size: 64
input [{ name: "input_ids" data_type: TYPE_INT64 dims: [-1] }]
output [{ name: "logits" data_type: TYPE_FP32 dims: [-1, -1] }]
instance_group [{ count: 1 kind: KIND_GPU gpus: [0] }]
optimization {
  execution_accelerators {
    gpu_execution_accelerator: [{ name: "tensorrt" }]  # TRT optimization
  }
}
```

## Pickle Security

**Pickle files can execute arbitrary code on load.** A malicious `.pt` or `.bin` file can:
- Execute shell commands
- Exfiltrate data
- Install backdoors
- Modify model weights silently

### Mitigations

1. **Never load pickle from untrusted sources** — always use SafeTensors when available
2. **`weights_only=True`** (PyTorch 2.0+):
   ```python
   state_dict = torch.load("model.pt", weights_only=True, map_location="cpu")
   ```
   Restricts unpickling to tensor types only. Default in PyTorch 2.6+.
3. **Scan before loading**: `huggingface_hub` runs pickle scanning on uploads
4. **Convert immediately**: If you must use a pickle checkpoint, convert to SafeTensors:
   ```python
   import torch
   from safetensors.torch import save_file
   state_dict = torch.load("model.pt", weights_only=True)
   save_file(state_dict, "model.safetensors")
   ```

## Conversion Matrix

| From → To | Tool / Method |
|-----------|--------------|
| SafeTensors → GGUF | `convert_hf_to_gguf.py` (llama.cpp) |
| Pickle → SafeTensors | `torch.load()` + `safetensors.torch.save_file()` |
| Pickle → GGUF | Convert to SafeTensors first, then GGUF |
| SafeTensors → ONNX | `torch.onnx.export()` or `optimum` CLI |
| ONNX → TensorRT | `trtexec --onnx=model.onnx --saveEngine=model.engine` |
| GGUF → SafeTensors | Not standard — dequantize with llama.cpp, re-save |
| HF Hub → Ollama | `FROM hf.co/<org>/<model>-GGUF:<quant>` in Modelfile |

## Framework Compatibility

| Format | vLLM | HF Transformers | Triton | Ollama | llama.cpp | TGI |
|--------|------|----------------|--------|--------|-----------|-----|
| SafeTensors | ✅ Primary | ✅ Primary | ✅ (Python backend) | ✅ (auto-convert) | ✅ (via convert) | ✅ |
| GGUF | ❌ | ❌ | ❌ | ✅ Primary | ✅ Primary | ✅ (limited) |
| ONNX | ❌ | ✅ (via Optimum) | ✅ Primary | ❌ | ❌ | ❌ |
| Pickle | ✅ (legacy) | ✅ (legacy) | ✅ (Python backend) | ❌ | ❌ | ✅ |
| TensorRT | ❌ | ❌ | ✅ (TRT backend) | ❌ | ❌ | ❌ |

## Choosing a Format

**Training/Fine-tuning**: SafeTensors. Always. Memory-mapped, safe, fast, universal support.

**GPU Inference (datacenter)**: SafeTensors → load into vLLM, TGI, or Triton. ONNX if you need cross-framework or TensorRT optimization.

**CPU/Edge Inference**: GGUF with appropriate quantization (Q4_K_M is the common default). Ollama or llama.cpp.

**Archival/Distribution**: SafeTensors on Hugging Face Hub. GGUF alongside for quantized variants.

## Cross-References

- [vllm](../vllm/) — SafeTensors loading, `load_format` configuration
- [huggingface-transformers](../huggingface-transformers/) — SafeTensors as primary format on HF Hub
- [ollama](../ollama/) — GGUF model loading and Modelfile configuration
- [flash-attention](../flash-attention/) — Attention backends used after model loading

## Reference

- [SafeTensors GitHub](https://github.com/huggingface/safetensors)
- [GGUF spec](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [ONNX spec](https://onnx.ai/onnx/intro/)
- [HuggingFace Optimum](https://huggingface.co/docs/optimum/)
- `scripts/convert_hf_to_gguf.sh` — convert HuggingFace model to GGUF with optional quantization
