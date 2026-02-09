---
name: ollama
description: Ollama local LLM serving — Modelfile configuration, CLI, model management, GPU backends (CUDA, Vulkan), API, and runtime tuning. Use when configuring Ollama for LLM inference, creating Modelfiles, managing models, selecting GPU backends, or tuning runtime performance.
---

# Ollama

## Architecture

Ollama wraps llama.cpp with a model registry, REST API, and runtime management. It pulls GGUF models from the Ollama library or Hugging Face, manages them locally, and serves them via an OpenAI-compatible API on port 11434.

## CLI

### Model Management

```bash
# Pull models (tag format: name:variant)
ollama pull llama3.2:3b
ollama pull qwen2.5:72b-instruct-q4_K_M
ollama pull nomic-embed-text              # Embedding model

# List local models (name, size, quantization, modified date)
ollama list

# Show model details — parameters, template, license, system prompt
ollama show llama3.2:3b
ollama show llama3.2:3b --modelfile       # Dump as Modelfile
ollama show llama3.2:3b --parameters      # Just parameters
ollama show llama3.2:3b --template        # Just chat template
ollama show llama3.2:3b --license         # Just license
ollama show llama3.2:3b --system          # Just system prompt

# Running models (loaded in VRAM — shows VRAM usage, processor, expiry)
ollama ps

# Interactive chat
ollama run llama3.2:3b
ollama run llama3.2:3b "What is K8s?"     # Single prompt, non-interactive

# Copy/tag model (for creating variants)
ollama cp llama3.2:3b my-llama:latest

# Remove model
ollama rm my-assistant

# Create from Modelfile
ollama create my-assistant -f Modelfile

# Push to Ollama registry (requires ollama.com account)
ollama push myuser/my-assistant

# Quantize an existing model
ollama create my-model-q4 --quantize q4_K_M -f Modelfile
```

### Server Management

```bash
# Start server (foreground)
ollama serve

# Check running server
ollama ps

# List available commands
ollama help
```

### Useful CLI Patterns

```bash
# Pipe input
echo "Summarize this" | ollama run llama3.2:3b

# JSON output mode
ollama run llama3.2:3b "List 3 fruits as JSON" --format json

# Set system prompt inline
ollama run llama3.2:3b --system "You are a K8s expert"

# Preload model into memory without running
curl http://localhost:11434/api/generate -d '{"model":"llama3.2:3b","keep_alive":"1h"}'

# Unload model from memory
curl http://localhost:11434/api/generate -d '{"model":"llama3.2:3b","keep_alive":"0"}'
```

## Thinking / Reasoning

Models with reasoning capabilities (e.g., `qwen3`, `deepseek-r1`) can expose their chain-of-thought via the `think` parameter:

```python
from ollama import chat

# Non-streaming
response = chat(
    model='qwen3',
    messages=[{'role': 'user', 'content': 'How many r in strawberry?'}],
    think=True,
    stream=False,
)
print('Thinking:', response.message.thinking)
print('Answer:', response.message.content)
```

```bash
# CLI: thinking enabled by default for supported models
ollama run qwen3 "What is 17 x 23?"

# Disable thinking with /nothink, re-enable with /think
```

```bash
# API: set "think": true in the request
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3",
  "messages": [{"role": "user", "content": "How many r in strawberry?"}],
  "think": true,
  "stream": false
}'
# Response includes message.thinking (reasoning trace) and message.content (final answer)
```

The `thinking` field is separate from `content` — clients can show/hide the reasoning trace independently. For budget control, some models accept think levels (`"think": "low"`, `"medium"`, `"high"`) instead of boolean.

## Modelfile

A Modelfile defines a model's base weights, system prompt, and generation parameters:

```dockerfile
# Base model — required
FROM llama3.2:3b

# System prompt
SYSTEM """You are a helpful coding assistant specializing in Python and Kubernetes."""

# Generation parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_predict 4096
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|eot_id|>"
PARAMETER num_ctx 8192
PARAMETER num_gpu 99
PARAMETER num_thread 8

# Chat template (Go template syntax)
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>
{{ .Response }}<|eot_id|>"""

# Adapter (LoRA/QLoRA fine-tune)
ADAPTER ./lora-adapter.gguf

# License
LICENSE "Apache-2.0"
```

### All PARAMETER Options

| Parameter | Type | Default | Effect |
|-----------|------|---------|--------|
| `temperature` | float | 0.8 | Sampling temperature (0 = greedy) |
| `top_p` | float | 0.9 | Nucleus sampling threshold |
| `top_k` | int | 40 | Top-k sampling (0 = disabled) |
| `num_predict` | int | -1 | Max tokens to generate (-1 = infinite, -2 = fill context) |
| `repeat_penalty` | float | 1.1 | Penalize repeated tokens |
| `repeat_last_n` | int | 64 | Window for repeat penalty (0 = disabled, -1 = num_ctx) |
| `num_ctx` | int | 2048 | Context window size |
| `num_gpu` | int | varies | Layers to offload to GPU (0 = CPU only, 99 = all) |
| `num_thread` | int | auto | CPU threads for computation |
| `seed` | int | 0 | RNG seed (0 = random) |
| `stop` | string | — | Stop sequence (can specify multiple PARAMETER stop lines) |
| `tfs_z` | float | 1.0 | Tail-free sampling (1.0 = disabled) |
| `typical_p` | float | 1.0 | Locally typical sampling (1.0 = disabled) |
| `mirostat` | int | 0 | Mirostat mode (0 = disabled, 1, 2) |
| `mirostat_tau` | float | 5.0 | Target entropy for Mirostat |
| `mirostat_eta` | float | 0.1 | Learning rate for Mirostat |
| `penalize_newline` | bool | true | Penalize newlines in repeat penalty |
| `numa` | bool | false | Enable NUMA optimization |

### FROM Sources

```dockerfile
# Ollama library
FROM llama3.2:3b
FROM qwen2.5:72b-instruct-q4_K_M

# Local GGUF file
FROM ./models/custom-model-Q4_K_M.gguf

# Hugging Face Hub (auto-converts SafeTensors → GGUF)
FROM hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M

# SafeTensors from HF (auto-conversion)
FROM hf.co/meta-llama/Llama-3.2-3B-Instruct
```

### Modelfile Instructions Reference

| Instruction | Required | Effect |
|------------|----------|--------|
| `FROM` | Yes | Base model (library tag, local GGUF, or HF path) |
| `PARAMETER` | No | Set generation parameters (see table above) |
| `TEMPLATE` | No | Override chat template (Go template syntax) |
| `SYSTEM` | No | Default system prompt |
| `ADAPTER` | No | LoRA/QLoRA adapter path (GGUF format) |
| `LICENSE` | No | License text or identifier |
| `MESSAGE` | No | Pre-seed conversation history (`MESSAGE user "..."`, `MESSAGE assistant "..."`) |

## GPU Backends

### NVIDIA (CUDA)

Default backend for NVIDIA GPUs. Compute capability 5.0+ required (Maxwell and newer). No additional configuration needed — Ollama auto-detects CUDA GPUs.

```bash
# Select specific GPUs
CUDA_VISIBLE_DEVICES=0,1 ollama serve

# Force CPU only
CUDA_VISIBLE_DEVICES=-1 ollama serve
```

### Vulkan (Experimental)

Vulkan provides GPU acceleration for GPUs without dedicated CUDA/ROCm support — including Intel GPUs and AMD GPUs not covered by ROCm. Available since Ollama 0.12.6.

```bash
# Enable Vulkan backend
OLLAMA_VULKAN=1 ollama serve

# Select specific Vulkan GPU(s)
OLLAMA_VULKAN=1 GGML_VK_VISIBLE_DEVICES=0 ollama serve

# Disable Vulkan GPUs (force other backends)
GGML_VK_VISIBLE_DEVICES=-1 ollama serve
```

**Requirements**:
- Vulkan drivers installed (most Windows GPU drivers include Vulkan; Linux may need `mesa-vulkan-drivers` or vendor-specific packages)
- For Intel on Linux: install via [Intel GPU docs](https://dgpu-docs.intel.com/driver/client/overview.html)

**VRAM reporting**: Vulkan requires `cap_perfmon` capability or root to expose VRAM data. Without it, Ollama uses approximate model sizes for scheduling:

```bash
sudo setcap cap_perfmon+ep /usr/local/bin/ollama
```

**GPU selection env vars by backend**:

| Backend | Variable | Example |
|---------|----------|---------|
| NVIDIA (CUDA) | `CUDA_VISIBLE_DEVICES` | `0,1` or UUID |
| Vulkan | `GGML_VK_VISIBLE_DEVICES` | `0` (numeric ID) |

## Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `OLLAMA_HOST` | `127.0.0.1:11434` | Bind address |
| `OLLAMA_MODELS` | `~/.ollama/models` | Model storage path |
| `OLLAMA_NUM_PARALLEL` | `1` | Concurrent request slots per model |
| `OLLAMA_MAX_LOADED_MODELS` | `1` | Max models in VRAM simultaneously |
| `OLLAMA_KEEP_ALIVE` | `5m` | Default keep-alive duration |
| `OLLAMA_MAX_QUEUE` | `512` | Max queued requests |
| `OLLAMA_FLASH_ATTENTION` | `false` | Enable flash attention |
| `OLLAMA_KV_CACHE_TYPE` | `f16` | KV cache quantization (`f16`, `q8_0`, `q4_0`) |
| `OLLAMA_GPU_OVERHEAD` | `0` | Reserve GPU memory (bytes) for other processes |
| `OLLAMA_CONTEXT_LENGTH` | `2048` | Default context length (overrides model default) |
| `OLLAMA_LOAD_TIMEOUT` | `5m` | Timeout for model loading |
| `OLLAMA_NOPRUNE` | `0` | Don't prune old model blobs |
| `OLLAMA_DEBUG` | `0` | Enable debug logging |
| `OLLAMA_ORIGINS` | — | Allowed CORS origins |
| `OLLAMA_VULKAN` | `0` | Enable Vulkan GPU backend (experimental) |
| `OLLAMA_LLM_LIBRARY` | — | Override llama.cpp backend library selection |

## API

### OpenAI-Compatible Endpoints

```
POST /v1/chat/completions    — Chat completions
POST /v1/completions         — Text completions
POST /v1/embeddings          — Embeddings
GET  /v1/models              — List models
```

### Native Ollama API

```
POST /api/generate    — Raw text generation
POST /api/chat        — Chat with message history
POST /api/embed       — Generate embeddings (single or batch)
GET  /api/tags        — List local models
POST /api/show        — Model details (parameters, template, license)
POST /api/pull        — Pull model from registry
POST /api/push        — Push model to registry
DELETE /api/delete    — Delete model
POST /api/copy        — Copy/tag model
GET  /api/ps          — Running models (loaded in memory)
POST /api/create      — Create model from Modelfile
```

### Runtime Options (per-request overrides)

```json
{
  "model": "llama3.2:3b",
  "prompt": "Hello",
  "options": {
    "num_ctx": 4096,
    "temperature": 0.0,
    "num_gpu": 99,
    "num_predict": 512
  },
  "keep_alive": "5m"
}
```

`keep_alive` controls how long a model stays loaded after a request. `"0"` = unload immediately, `"-1"` = keep loaded indefinitely.

### OpenAI SDK Compatibility

Point any OpenAI SDK client at Ollama:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="unused",  # Ollama doesn't require auth
)

response = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Runtime Tuning

### Multi-Model Serving

```bash
# Load multiple models concurrently
OLLAMA_MAX_LOADED_MODELS=3 OLLAMA_NUM_PARALLEL=4 ollama serve
```

**Memory estimation**: Each loaded model consumes approximately its GGUF file size in VRAM, plus KV cache overhead per concurrent slot (`num_ctx × num_parallel × ~2MB` for fp16 KV).

### KV Cache Quantization

Reduce VRAM usage by quantizing the KV cache:

```bash
OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve   # ~50% KV VRAM reduction
OLLAMA_KV_CACHE_TYPE=q4_0 ollama serve   # ~75% KV VRAM reduction, some quality loss
```

### Flash Attention

Enable for faster attention computation and reduced memory:

```bash
OLLAMA_FLASH_ATTENTION=1 ollama serve
```

### Context Length

Override the model's default context window:

```bash
OLLAMA_CONTEXT_LENGTH=32768 ollama serve  # Global default
```

Or per-request via `options.num_ctx`, or per-model via `PARAMETER num_ctx` in Modelfile.

## Kubernetes Deployment

For K8s deployment, the key configuration points are the env vars above set on the container spec, a PVC for model storage at `/root/.ollama`, and GPU resource requests. A community Helm chart is available:

```bash
helm repo add ollama-helm https://otwld.github.io/ollama-helm/
helm install ollama ollama-helm/ollama \
  --set ollama.gpu.enabled=true \
  --set ollama.gpu.type=nvidia \
  --set ollama.gpu.number=1 \
  --set ollama.models.pull={llama3.2:3b} \
  --set persistentVolume.enabled=true \
  -n ollama --create-namespace
```

For init container model pre-pulling, mount the PVC and run `ollama serve & sleep 5 && ollama pull <model> && kill %1`.

## Ollama vs vLLM

| Feature | Ollama | vLLM |
|---------|--------|------|
| Target | Local/edge inference | Datacenter high-throughput |
| Quantization | GGUF (Q2–Q8, IQ) | AWQ, GPTQ, FP8 |
| Batching | Basic (num_parallel) | Continuous batching |
| Multi-model | Built-in (hot-swap) | One model per process |
| GPU backends | CUDA, Vulkan, ROCm, Metal | CUDA, ROCm |
| Setup complexity | Minimal | Higher |

Use Ollama for development, prototyping, and single-user inference. Use vLLM for production serving with high concurrency.

## Cross-References

- [model-formats](../model-formats/) — GGUF format details, quantization types, SafeTensors conversion
- [vllm](../vllm/) — High-throughput alternative for datacenter inference
- [openai-api](../openai-api/) — OpenAI-compatible API patterns (Ollama implements the same API)
- [llm-evaluation](../llm-evaluation/) — Use Ollama as an evaluation backend
- [huggingface-transformers](../huggingface-transformers/) — Source models to convert to GGUF for Ollama

## Reference

- [Ollama docs](https://github.com/ollama/ollama/tree/main/docs)
- [Ollama API reference](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Modelfile reference](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)
- [Ollama model library](https://ollama.com/library)
