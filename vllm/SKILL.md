---
name: vllm
description: >
  Configure and use vLLM for high-throughput LLM inference. Use when:
  (1) Configuring vLLM engine settings (tensor parallelism, quantization, GPU memory, context length),
  (2) Using the OpenAI-compatible API (completions, chat, embeddings),
  (3) Serving with LoRA adapters, speculative decoding, or structured output,
  (4) Tuning production settings (continuous batching, prefix caching, chunked prefill),
  (5) Multi-GPU/multi-node inference, (6) Debugging OOM, slow TTFT, or throughput issues.
  Assumes the user knows how to run vLLM — focuses on configuration and settings.
---

# vLLM

vLLM is a high-throughput LLM inference engine using PagedAttention for optimal GPU memory management. Version: **0.8+** (docs target current stable).

## Engine Configuration

All settings can be passed as Python kwargs to `LLM()` (offline) or `AsyncLLM.from_engine_args()` (server), or as CLI flags to `vllm serve`. This document uses **kwargs** form.

> **V1 Engine**: vLLM V1 is now the default engine. Key differences from V0: chunked prefill enabled by default, different logprobs semantics (raw by default, use `--logprobs-mode` to change), higher CUDA graph memory usage. Set `VLLM_USE_V1=0` to fall back to V0 if needed.

### Core Settings

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    dtype="bfloat16",                  # auto, float16, bfloat16, float32
    tensor_parallel_size=4,            # shard across 4 GPUs
    gpu_memory_utilization=0.90,       # fraction of GPU memory for KV cache
    max_model_len=8192,                # max context length
    max_num_seqs=256,                  # max concurrent sequences
    enable_prefix_caching=True,        # reuse KV cache for shared prefixes
    enable_chunked_prefill=True,       # overlap prefill and decode
    trust_remote_code=False,           # allow model custom code
)
```

### Full Configuration Reference

| Kwarg | Purpose | Default |
|-------|---------|---------|
| `model` | HuggingFace model ID or local path | required |
| `dtype` | Weight dtype (`auto`, `float16`, `bfloat16`, `float32`) | `auto` |
| `tensor_parallel_size` | Number of GPUs for tensor parallelism | `1` |
| `pipeline_parallel_size` | Number of stages for pipeline parallelism | `1` |
| `gpu_memory_utilization` | Fraction of GPU memory for KV cache | `0.9` |
| `max_model_len` | Maximum sequence length | Model's max |
| `max_num_seqs` | Max concurrent sequences (batch size) | `256` |
| `max_num_batched_tokens` | Max tokens per scheduler iteration | Auto |
| `quantization` | Quantization method (`gptq`, `awq`, `fp8`, `bitsandbytes`) | `None` |
| `enable_prefix_caching` | Automatic prefix caching (APC) | `False` |
| `enable_chunked_prefill` | Chunked prefill scheduling | `True` (V1) |
| `enforce_eager` | Disable CUDA graphs (for debugging) | `False` |
| `trust_remote_code` | Allow model custom code | `False` |
| `served_model_name` | Override model name in API | Model ID |
| `chat_template` | Jinja2 chat template path | From tokenizer |
| `seed` | Random seed for reproducibility | `0` |
| `disable_log_requests` | Reduce logging overhead | `False` |

### Scheduler Settings

```python
llm = LLM(
    model="...",
    scheduling_policy="fcfs",          # "fcfs" (default) or "priority"
    max_num_seqs=256,                  # max concurrent sequences
    max_num_batched_tokens=32768,      # max tokens per scheduler step
    num_scheduler_steps=1,             # >1 = multi-step scheduling (amortize overhead)
    preemption_mode="recompute",       # "recompute" (faster) or "swap" (saves GPU)
)
```

### KV Cache Settings

```python
llm = LLM(
    model="...",
    gpu_memory_utilization=0.90,       # fraction of GPU memory for KV cache (after model)
    kv_cache_dtype="auto",             # "auto", "fp8", "fp8_e5m2", "fp8_e4m3"
    block_size=16,                     # KV cache block size (16 or 32)
    swap_space=4,                      # GiB of CPU swap for preempted sequences
    cpu_offload_gb=0,                  # offload model weights to CPU (reduces GPU usage)
    num_gpu_blocks_override=None,      # manually set GPU KV blocks (for testing)
)
```

### Tokenizer & Chat Settings

```python
llm = LLM(
    model="...",
    tokenizer=None,                    # override tokenizer (different from model)
    chat_template="/path/template.jinja",  # custom Jinja2 chat template
    tokenizer_mode="auto",             # "auto" or "slow" (slow = no fast tokenizer)
    max_logprobs=20,                   # max logprobs returnable per token
    tokenizer_pool_size=0,             # >0 = async tokenizer workers
    tokenizer_pool_type="ray",         # "ray" (for distributed tokenizer)
)
```

### Model Loading Settings

```python
llm = LLM(
    model="...",
    load_format="auto",                # "auto", "pt", "safetensors", "npcache", "dummy", "bitsandbytes"
    download_dir=None,                 # custom download directory
    model_loader_extra_config=None,    # dict of extra config for model loader
    revision="main",                   # model revision/branch
    code_revision="main",              # code revision for trust_remote_code models
    config_format="auto",              # "auto", "hf", or "mistral"
)
```

## Quantization

Reduce memory footprint and increase throughput:

```python
# GPTQ (4-bit, pre-quantized models)
llm = LLM(model="TheBloke/Llama-2-70B-Chat-GPTQ", quantization="gptq", tensor_parallel_size=2)

# AWQ (4-bit, fast)
llm = LLM(model="TheBloke/Llama-2-70B-Chat-AWQ", quantization="awq")

# FP8 (8-bit, minimal quality loss — A100/H100)
llm = LLM(model="meta-llama/Llama-3.1-70B-Instruct", quantization="fp8")

# bitsandbytes (dynamic quantization, no pre-quantized model needed)
llm = LLM(model="meta-llama/Llama-3.1-70B-Instruct", quantization="bitsandbytes", load_format="bitsandbytes")
```

| Method | Bits | Pre-quantized? | Quality | Speed | Best For |
|--------|------|---------------|---------|-------|----------|
| FP8 | 8 | No | Excellent | Fast | A100/H100 production |
| GPTQ | 4 | Yes | Good | Fast | Memory-constrained |
| AWQ | 4 | Yes | Good | Fastest 4-bit | Throughput-focused |
| bitsandbytes | 4/8 | No | Good | Slower | Quick experiments |

## Sampling Parameters

```python
params = SamplingParams(
    temperature=0.7,                   # 0.0 = greedy
    top_p=0.9,
    top_k=50,                          # -1 = disabled
    max_tokens=512,
    min_tokens=0,                      # minimum before stop
    frequency_penalty=0.0,
    presence_penalty=0.0,
    repetition_penalty=1.0,
    stop=["<|end|>", "\n\n"],          # stop sequences
    n=1,                               # number of completions
    best_of=None,                      # sample best_of, return n
    logprobs=None,                     # number of logprobs per token
    prompt_logprobs=None,              # logprobs for prompt tokens
    skip_special_tokens=True,
    spaces_between_special_tokens=True,
    seed=None,                         # per-request seed
)
```

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

### Completions (Legacy)

```python
response = client.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    prompt="The key advantage of PagedAttention is",
    max_tokens=100,
    temperature=0.0,
)
```

### Embeddings

```python
# Start server with: LLM(model="BAAI/bge-large-en-v1.5", task="embed")
response = client.embeddings.create(
    model="BAAI/bge-large-en-v1.5",
    input=["What is vLLM?", "How does PagedAttention work?"],
)
```

## Advanced Features

### LoRA Adapters

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_lora=True,
    lora_modules={
        "code-adapter": "./adapters/code-lora",
        "chat-adapter": "./adapters/chat-lora",
    },
    max_lora_rank=64,
    max_loras=4,                       # max loaded simultaneously
    lora_extra_vocab_size=256,         # extra vocab for adapter tokens
)

# Request a specific adapter via API
response = client.chat.completions.create(
    model="code-adapter",
    messages=[{"role": "user", "content": "Write a Python sort function"}],
)
```

### Structured Output

```python
from pydantic import BaseModel

class ModelEval(BaseModel):
    model_name: str
    accuracy: float
    parameters_millions: int
    recommended: bool

# Preferred: response_format with json_schema
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Evaluate BERT-base for text classification."}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "model-eval",
            "schema": ModelEval.model_json_schema(),
        },
    },
)

# Alternative: structured_outputs in extra_body (regex, choice, grammar)
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Classify sentiment."}],
    extra_body={
        "structured_outputs": {"choice": ["positive", "negative", "neutral"]},
        # Or: "structured_outputs": {"regex": r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"},
    },
)
```

> **Deprecated**: `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar` in `extra_body` still work but are deprecated. Migrate to `response_format` (for JSON schema) or `structured_outputs` (for regex/choice/grammar). The `guided_decoding_backend` request parameter has been removed.

### Speculative Decoding

```python
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    speculative_model="meta-llama/Llama-3.2-1B-Instruct",
    num_speculative_tokens=5,
)
```

### Prefix Caching

```python
llm = LLM(model="...", enable_prefix_caching=True)
```

Effective for: shared system prompts, RAG with recurring chunks, multi-turn conversations. Incompatible with sliding window attention models.

For **cache isolation** in shared environments, include `"cache_salt": "your-salt"` in the request body — only requests with the same salt can reuse cached KV blocks.

### Chunked Prefill

```python
llm = LLM(
    model="...",
    enable_chunked_prefill=True,
    max_num_batched_tokens=2048,       # chunk size; smaller = lower TTFT variance
)
```

Reduces head-of-line blocking from long prompts. Essential for mixed-length workloads.

### Vision / Multimodal Models

```python
llm = LLM(
    model="Qwen/Qwen2-VL-7B-Instruct",
    max_model_len=8192,
    limit_mm_per_prompt={"image": 4},  # max images per request
)

# API usage
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

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_auto_tool_choice=True,
    tool_call_parser="hermes",         # "hermes", "llama3_json", "mistral", etc.
)
```

### Disaggregated Prefill-Decode Serving

Separate prefill and decode phases onto different vLLM instances for independent TTFT/ITL tuning. See `references/disaggregated-serving.md` for full architecture, all connector types (NixlConnector, LMCacheConnectorV1, P2pNcclConnector, OffloadingConnector, MultiConnector), and K8s deployment patterns.

```python
from vllm.config import KVTransferConfig

# Prefill instance
llm = LLM(
    model="...",
    tensor_parallel_size=4,
    kv_transfer_config=KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_producer",
    ),
)
```

### LMCache (KV Cache Sharing)

Share KV caches across vLLM replicas to avoid redundant prefill computation. See `references/lmcache.md` for configuration, K8s deployment, and CPU offloading.

### Alternative Model Loading (Run:ai Streamer)

Stream model weights directly from S3/local storage with configurable concurrency:

```python
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    load_format="runai_streamer",
    model_loader_extra_config={
        "concurrency": 16,              # parallel file read threads / S3 clients
        "memory_limit": 0,              # CPU memory limit (0 = unlimited)
    },
)

# For pre-sharded models (faster multi-GPU loading)
llm = LLM(
    model="/path/to/sharded/model",
    load_format="runai_streamer_sharded",
    model_loader_extra_config={
        "concurrency": 16,
        "distributed": True,            # each GPU loads its own shard
    },
)
```

Supports loading from: local filesystem, S3 (`s3://bucket/path`), and other object stores.

## Multi-GPU and Multi-Node

### Tensor Parallelism (Single Node)

```python
llm = LLM(model="meta-llama/Llama-3.1-70B-Instruct", tensor_parallel_size=4)
```

### Pipeline Parallelism

```python
# 8 GPUs total (4 TP × 2 PP)
llm = LLM(
    model="meta-llama/Llama-3.1-405B-Instruct",
    tensor_parallel_size=4,
    pipeline_parallel_size=2,
)
```

### Multi-Node with Ray

```python
llm = LLM(
    model="meta-llama/Llama-3.1-405B-Instruct",
    tensor_parallel_size=8,
    distributed_executor_backend="ray",
)
# Requires Ray cluster running: ray start --head on head node, ray start --address=head:6379 on workers
```

## Environment Variables

| Variable | Purpose |
|---|---|
| `VLLM_ATTENTION_BACKEND` | Override attention backend: `FLASH_ATTN`, `XFORMERS`, `FLASHINFER` |
| `VLLM_USE_V1` | V1 engine is default; set `0` to fall back to V0 |
| `CUDA_VISIBLE_DEVICES` | Restrict visible GPUs |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` or `fork` for worker processes |
| `VLLM_PP_LAYER_PARTITION` | Custom pipeline parallel layer splits (e.g., `10,22`) |
| `VLLM_LOGGING_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `HF_TOKEN` | Hugging Face token for gated models |

## Performance Tuning

### Throughput vs Latency

| Setting | Throughput ↑ | Latency ↓ |
|---------|-------------|-----------|
| `max_num_seqs` | Increase (512+) | Decrease (32-64) |
| `enable_chunked_prefill` | ✅ | Mixed |
| `enable_prefix_caching` | ✅ (shared prefixes) | ✅ |
| `gpu_memory_utilization` | Increase (0.95) | — |
| `max_num_batched_tokens` | Increase | Decrease |
| Quantization (FP8/AWQ) | ✅ | ✅ |

### Metrics Endpoint

vLLM exposes Prometheus metrics at `/metrics`:

```
vllm:num_requests_running            # currently processing
vllm:num_requests_waiting            # in queue
vllm:num_requests_swapped            # swapped to CPU
vllm:gpu_cache_usage_perc            # KV cache utilization
vllm:avg_prompt_throughput_toks_per_s
vllm:avg_generation_throughput_toks_per_s
vllm:time_to_first_token_seconds     # TTFT histogram
vllm:time_per_output_token_seconds   # TPOT histogram
```

## Debugging

See `references/troubleshooting.md` for:
- GPU OOM during model loading or inference
- Slow time-to-first-token (TTFT)
- Throughput degradation under load
- Model loading failures and multi-GPU communication errors
- LoRA, structured output, speculative decoding, and multimodal issues

## Cross-References

- [openai-api](../openai-api/) — OpenAI-compatible API served by vLLM
- [model-formats](../model-formats/) — SafeTensors, GGUF, and model format details
- [flash-attention](../flash-attention/) — Attention backends used by vLLM
- [ollama](../ollama/) — Lightweight alternative for local inference
- [ray-serve](../ray-serve/) — Deploy vLLM behind Ray Serve for autoscaling
- [llm-evaluation](../llm-evaluation/) — Use vLLM backend for LLM benchmarking
- [kuberay](../kuberay/) — Deploy vLLM on Ray clusters via KubeRay

## Reference

- [vLLM docs](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [Supported models](https://docs.vllm.ai/en/latest/models/supported_models/)
- [Engine arguments reference](https://docs.vllm.ai/en/latest/serving/engine_args.html)
- `references/troubleshooting.md` — common errors and fixes
- `references/disaggregated-serving.md` — PD separation architecture and connectors
- `references/lmcache.md` — KV cache sharing and CPU offloading
