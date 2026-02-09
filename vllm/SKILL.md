---
name: vllm
description: >
  Deploy and configure vLLM for high-throughput LLM inference and serving. Use when:
  (1) Deploying vLLM as an inference server (Docker, Kubernetes, bare metal), (2) Configuring
  engine settings (tensor parallelism, quantization, GPU memory, max model length),
  (3) Using the OpenAI-compatible API (completions, chat, embeddings), (4) Serving with
  LoRA adapters, speculative decoding, or structured output, (5) Tuning production settings
  (continuous batching, prefix caching, chunked prefill), (6) Deploying on Kubernetes with
  autoscaling, (7) Multi-GPU/multi-node inference, (8) Debugging OOM, slow TTFT, or
  throughput issues.
---

# vLLM

vLLM is a high-throughput LLM inference engine using PagedAttention for optimal GPU memory management. Version: **0.8.x+**.

## Quick Start

### Bare Metal

```bash
pip install vllm

# Start server
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --dtype auto \
  --api-key token-abc123 \
  --port 8000
```

### Docker

```bash
docker run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dtype auto \
  --api-key token-abc123
```

### Python API (Offline Inference)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", gpu_memory_utilization=0.9)
params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)

outputs = llm.generate(["Explain PagedAttention in one paragraph."], params)
print(outputs[0].outputs[0].text)
```

## Engine Configuration

### Key Server Arguments

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \           # shard across 4 GPUs
  --gpu-memory-utilization 0.90 \      # % of GPU memory for KV cache
  --max-model-len 8192 \              # max context length
  --max-num-seqs 256 \                # max concurrent sequences
  --enable-prefix-caching \           # reuse KV cache for shared prefixes
  --enable-chunked-prefill \          # overlap prefill and decode
  --disable-log-requests \            # reduce logging overhead
  --api-key $VLLM_API_KEY
```

### Configuration Reference

| Argument | Purpose | Default |
|----------|---------|---------|
| `--model` | HuggingFace model ID or local path | required |
| `--dtype` | Weight dtype (auto, float16, bfloat16, float32) | `auto` |
| `--tensor-parallel-size` | Number of GPUs for tensor parallelism | `1` |
| `--pipeline-parallel-size` | Number of stages for pipeline parallelism | `1` |
| `--gpu-memory-utilization` | Fraction of GPU memory for KV cache | `0.9` |
| `--max-model-len` | Maximum sequence length | Model's max |
| `--max-num-seqs` | Max concurrent sequences (batch size) | `256` |
| `--max-num-batched-tokens` | Max tokens per iteration | Auto |
| `--quantization` | Quantization method | None |
| `--enable-prefix-caching` | Automatic prefix caching (APC) | `false` |
| `--enable-chunked-prefill` | Chunked prefill scheduling | `false` |
| `--enforce-eager` | Disable CUDA graphs (debug) | `false` |
| `--trust-remote-code` | Allow model custom code | `false` |
| `--served-model-name` | Override model name in API | Model ID |
| `--chat-template` | Jinja2 chat template path | From tokenizer |
| `--host` | Bind address | `0.0.0.0` |
| `--port` | Bind port | `8000` |

## Quantization

Reduce memory footprint and increase throughput:

```bash
# GPTQ (4-bit, pre-quantized models)
vllm serve TheBloke/Llama-2-70B-Chat-GPTQ \
  --quantization gptq \
  --tensor-parallel-size 2

# AWQ (4-bit, fast)
vllm serve TheBloke/Llama-2-70B-Chat-AWQ \
  --quantization awq

# FP8 (8-bit, minimal quality loss — A100/H100)
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --quantization fp8

# bitsandbytes (dynamic quantization, no pre-quantized model needed)
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --quantization bitsandbytes \
  --load-format bitsandbytes
```

| Method | Bits | Pre-quantized? | Quality | Speed | Best For |
|--------|------|---------------|---------|-------|----------|
| FP8 | 8 | No | Excellent | Fast | A100/H100 production |
| GPTQ | 4 | Yes | Good | Fast | Memory-constrained |
| AWQ | 4 | Yes | Good | Fastest 4-bit | Throughput-focused |
| bitsandbytes | 4/8 | No | Good | Slower | Quick experiments |

## OpenAI-Compatible API

### Chat Completions

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-abc123")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful ML engineer."},
        {"role": "user", "content": "Explain LoRA fine-tuning."},
    ],
    temperature=0.7,
    max_tokens=512,
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Completions (Text)

```python
response = client.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    prompt="The key advantage of PagedAttention is",
    max_tokens=100,
    temperature=0.0,
)
```

### Embeddings

```bash
vllm serve BAAI/bge-large-en-v1.5 --task embed
```

```python
response = client.embeddings.create(
    model="BAAI/bge-large-en-v1.5",
    input=["What is vLLM?", "How does PagedAttention work?"],
)
```

## Advanced Features

### LoRA Adapters

Serve a base model with dynamically-loaded LoRA adapters:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-lora \
  --lora-modules \
    code-adapter=./adapters/code-lora \
    chat-adapter=./adapters/chat-lora \
  --max-lora-rank 64
```

```python
# Request a specific adapter
response = client.chat.completions.create(
    model="code-adapter",  # use the LoRA adapter name
    messages=[{"role": "user", "content": "Write a Python sort function"}],
)
```

### Structured Output (Guided Generation)

Force the model to output valid JSON or match a schema:

```python
from pydantic import BaseModel

class ModelEval(BaseModel):
    model_name: str
    accuracy: float
    parameters_millions: int
    recommended: bool

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Evaluate BERT-base for text classification."}],
    extra_body={
        "guided_json": ModelEval.model_schema_json(),
    },
)
# Response is guaranteed valid JSON matching the schema

# Or use regex
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Give me an IPv4 address"}],
    extra_body={
        "guided_regex": r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
    },
)

# Or constrain to choices
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Is this sentiment positive or negative?"}],
    extra_body={
        "guided_choice": ["positive", "negative", "neutral"],
    },
)
```

### Speculative Decoding

Use a small draft model to speed up generation:

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --speculative-model meta-llama/Llama-3.2-1B-Instruct \
  --num-speculative-tokens 5
```

### Prefix Caching

Reuses KV cache for requests sharing the same prefix (e.g., system prompt):

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-prefix-caching
```

Particularly effective for:
- Shared system prompts across requests
- RAG with recurring document chunks
- Multi-turn conversations (prior turns cached)

### Vision / Multimodal Models

```bash
vllm serve Qwen/Qwen2-VL-7B-Instruct \
  --max-model-len 8192 \
  --limit-mm-per-prompt image=4   # max images per request
```

```python
# API usage with images
response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
            {"type": "text", "text": "Describe this image."},
        ],
    }],
)
```

### Tool Calling

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-auto-tool-choice \
  --tool-call-parser hermes     # or llama3_json, mistral, etc.
```

### Disaggregated Prefill (v1 Engine)

Separate prefill and decode phases across GPU pools for higher throughput:

```bash
vllm serve model \
  --enable-disagg-prefill \
  --prefill-tp 4 \
  --decode-tp 4
```

## Multi-GPU and Multi-Node

### Tensor Parallelism (Single Node)

Shard the model across GPUs on one machine:

```bash
# 4 GPUs on one node
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4
```

### Pipeline Parallelism

Split model layers across stages (useful when TP alone isn't enough):

```bash
vllm serve meta-llama/Llama-3.1-405B-Instruct \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2
# Uses 8 GPUs total (4 TP × 2 PP)
```

### Multi-Node with Ray

```bash
# On head node
ray start --head --port=6379

# On worker nodes
ray start --address=head-node:6379

# Launch vLLM (auto-detects Ray cluster)
vllm serve meta-llama/Llama-3.1-405B-Instruct \
  --tensor-parallel-size 8 \
  --distributed-executor-backend ray
```

## Kubernetes Deployment

### Basic Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          args:
            - --model=meta-llama/Llama-3.1-8B-Instruct
            - --dtype=bfloat16
            - --gpu-memory-utilization=0.9
            - --max-model-len=8192
            - --enable-prefix-caching
            - --port=8000
          ports:
            - containerPort: 8000
          env:
            - name: HUGGING_FACE_HUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token
                  key: token
          resources:
            limits:
              nvidia.com/gpu: "1"
            requests:
              cpu: "4"
              memory: "32Gi"
              nvidia.com/gpu: "1"
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 180
            periodSeconds: 30
          volumeMounts:
            - name: model-cache
              mountPath: /root/.cache/huggingface
            - name: shm
              mountPath: /dev/shm
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache-pvc
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 8Gi
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-server
spec:
  selector:
    app: vllm
  ports:
    - port: 8000
      targetPort: 8000
```

### Autoscaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-server
  minReplicas: 1
  maxReplicas: 4
  metrics:
    # Scale on GPU utilization (requires DCGM exporter + Prometheus adapter)
    - type: Pods
      pods:
        metric:
          name: DCGM_FI_DEV_GPU_UTIL
        target:
          type: AverageValue
          averageValue: "80"
    # Or scale on pending requests via custom metrics
    - type: Pods
      pods:
        metric:
          name: vllm_num_requests_waiting
        target:
          type: AverageValue
          averageValue: "10"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300  # slow scale-down (GPU pods are expensive to start)
```

## Performance Tuning

### Throughput vs Latency

| Setting | Throughput ↑ | Latency ↓ |
|---------|-------------|-----------|
| `--max-num-seqs` | Increase (512+) | Decrease (32-64) |
| `--enable-chunked-prefill` | ✅ | Mixed |
| `--enable-prefix-caching` | ✅ (shared prefixes) | ✅ |
| `--gpu-memory-utilization` | Increase (0.95) | - |
| `--max-num-batched-tokens` | Increase | Decrease |
| Quantization (FP8/AWQ) | ✅ | ✅ |

### Benchmarking

```bash
# Built-in benchmark tool
python -m vllm.entrypoints.openai.api_server &

# Benchmark throughput
python -m vllm.benchmark_throughput \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 512 \
  --output-len 128 \
  --num-prompts 1000

# Benchmark latency
python -m vllm.benchmark_latency \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 512 \
  --output-len 128 \
  --batch-size 32
```

### Metrics Endpoint

vLLM exposes Prometheus metrics at `/metrics`:

```
vllm:num_requests_running     # currently processing
vllm:num_requests_waiting     # in queue
vllm:num_requests_swapped     # swapped to CPU
vllm:gpu_cache_usage_perc     # KV cache utilization
vllm:cpu_cache_usage_perc
vllm:avg_prompt_throughput_toks_per_s
vllm:avg_generation_throughput_toks_per_s
vllm:time_to_first_token_seconds  # TTFT histogram
vllm:time_per_output_token_seconds  # TPOT histogram
```

## Debugging

See `references/troubleshooting.md` for:
- GPU OOM during model loading or inference
- Slow time-to-first-token (TTFT)
- Throughput degradation under load
- Model loading failures
- NCCL errors in multi-GPU setups

## Reference

- [vLLM docs](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [Supported models](https://docs.vllm.ai/en/latest/models/supported_models/)
- `references/troubleshooting.md` — common errors and fixes
