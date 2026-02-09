---
name: ollama
description: Ollama local LLM serving — Modelfile configuration, model management, API, and Kubernetes deployment. Use when deploying or configuring Ollama for local LLM inference, creating Modelfiles, managing models, or running Ollama on Kubernetes.
---

# Ollama

## Architecture

Ollama wraps llama.cpp with a model registry, REST API, and runtime management. It pulls GGUF models from the Ollama library or Hugging Face, manages them locally, and serves them via an OpenAI-compatible API on port 11434.

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

# Chat template (Jinja2)
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

## Model Management

```bash
# Pull models
ollama pull llama3.2:3b
ollama pull qwen2.5:72b-instruct-q4_K_M

# Create from Modelfile
ollama create my-assistant -f Modelfile

# List local models
ollama list

# Show model details (parameters, template, license)
ollama show llama3.2:3b

# Copy/tag model
ollama cp llama3.2:3b my-llama:latest

# Remove model
ollama rm my-assistant

# Running models (loaded in memory)
ollama ps
```

## API

Ollama serves an OpenAI-compatible API:

### Chat Completions (OpenAI-compatible)

```
POST /v1/chat/completions
```

```json
{
  "model": "llama3.2:3b",
  "messages": [{"role": "user", "content": "Hello"}],
  "temperature": 0.7,
  "stream": true
}
```

### Native API

```
POST /api/generate    — Raw text generation
POST /api/chat        — Chat with message history
POST /api/embed       — Generate embeddings
GET  /api/tags        — List models
POST /api/show        — Model details
POST /api/pull        — Pull model
POST /api/push        — Push to registry
DELETE /api/delete    — Delete model
POST /api/copy        — Copy model
GET  /api/ps          — Running models
```

### Runtime Options (per-request)

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

`keep_alive` controls how long a model stays loaded in memory after a request. Set to `"0"` to unload immediately, `"-1"` to keep loaded indefinitely.

## Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `OLLAMA_HOST` | `127.0.0.1:11434` | Bind address |
| `OLLAMA_MODELS` | `~/.ollama/models` | Model storage path |
| `OLLAMA_NUM_PARALLEL` | `1` | Concurrent request slots per model |
| `OLLAMA_MAX_LOADED_MODELS` | `1` | Max models in VRAM simultaneously |
| `OLLAMA_KEEP_ALIVE` | `5m` | Default keep-alive duration |
| `OLLAMA_MAX_QUEUE` | `512` | Max queued requests |
| `OLLAMA_FLASH_ATTENTION` | `0` | Enable flash attention (`1`) |
| `OLLAMA_KV_CACHE_TYPE` | `f16` | KV cache quantization (`f16`, `q8_0`, `q4_0`) |
| `OLLAMA_GPU_OVERHEAD` | `0` | Reserve GPU memory (bytes) for other processes |
| `OLLAMA_NOPRUNE` | `0` | Don't prune old model blobs |
| `OLLAMA_DEBUG` | `0` | Enable debug logging |
| `OLLAMA_ORIGINS` | — | Allowed CORS origins |

## Kubernetes Deployment

### Deployment with GPU

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ollama
  namespace: ollama
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ollama
  template:
    metadata:
      labels:
        app: ollama
    spec:
      containers:
        - name: ollama
          image: ollama/ollama:latest
          ports:
            - containerPort: 11434
              name: http
          env:
            - name: OLLAMA_HOST
              value: "0.0.0.0:11434"
            - name: OLLAMA_KEEP_ALIVE
              value: "10m"
            - name: OLLAMA_NUM_PARALLEL
              value: "4"
            - name: OLLAMA_FLASH_ATTENTION
              value: "1"
          resources:
            requests:
              memory: "8Gi"
              cpu: "2"
              nvidia.com/gpu: "1"
            limits:
              memory: "16Gi"
              nvidia.com/gpu: "1"
          volumeMounts:
            - name: models
              mountPath: /root/.ollama
      volumes:
        - name: models
          persistentVolumeClaim:
            claimName: ollama-models
---
apiVersion: v1
kind: Service
metadata:
  name: ollama
  namespace: ollama
spec:
  type: ClusterIP
  selector:
    app: ollama
  ports:
    - port: 11434
      targetPort: http
      name: http
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ollama-models
  namespace: ollama
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 100Gi
```

### Model Pre-pulling with Init Container

```yaml
initContainers:
  - name: pull-models
    image: ollama/ollama:latest
    command: ["/bin/sh", "-c"]
    args:
      - |
        ollama serve &
        sleep 5
        ollama pull llama3.2:3b
        ollama pull nomic-embed-text
        kill %1
    volumeMounts:
      - name: models
        mountPath: /root/.ollama
```

### Helm Chart

The community Helm chart (`otwld/ollama-helm`) provides a simpler deployment:

```yaml
# values.yaml
ollama:
  gpu:
    enabled: true
    type: nvidia
    number: 1
  models:
    pull:
      - llama3.2:3b
      - nomic-embed-text

persistentVolume:
  enabled: true
  size: 100Gi

resources:
  requests:
    memory: 8Gi
    cpu: 2000m
  limits:
    memory: 16Gi
```

```bash
helm repo add ollama-helm https://otwld.github.io/ollama-helm/
helm install ollama ollama-helm/ollama -f values.yaml -n ollama
```

### Multi-Model Serving

To serve multiple models concurrently, increase `OLLAMA_MAX_LOADED_MODELS` and ensure sufficient VRAM:

```yaml
env:
  - name: OLLAMA_MAX_LOADED_MODELS
    value: "3"
  - name: OLLAMA_NUM_PARALLEL
    value: "4"
  - name: OLLAMA_GPU_OVERHEAD
    value: "536870912"   # 512MB reserved for system
```

**Memory estimation**: Each loaded model consumes approximately its GGUF file size in VRAM (plus KV cache overhead per concurrent slot).

## OpenAI SDK Compatibility

Point any OpenAI SDK client at Ollama:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://ollama.ollama.svc:11434/v1",
    api_key="unused",  # Ollama doesn't require auth
)

response = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Ollama vs vLLM

| Feature | Ollama | vLLM |
|---------|--------|------|
| Target | Local/edge inference | Datacenter high-throughput |
| Quantization | GGUF (Q2–Q8, IQ) | AWQ, GPTQ, FP8 |
| Batching | Basic (num_parallel) | Continuous batching |
| Throughput | Lower | Higher |
| Memory efficiency | Good (quantized) | Better (PagedAttention) |
| Multi-model | Built-in (hot-swap) | One model per process |
| Setup complexity | Minimal | Higher |
| GPU utilization | Moderate | High |

Use Ollama for development, prototyping, and single-user inference. Use vLLM for production serving with high concurrency.

## Cross-References

- [model-formats](../model-formats/) — GGUF format details, quantization types, SafeTensors conversion
- [vllm](../vllm/) — High-throughput alternative for datacenter inference
- [openai-api](../openai-api/) — OpenAI-compatible API patterns (Ollama implements the same API)
