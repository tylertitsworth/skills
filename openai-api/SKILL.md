---
name: openai-api
description: >
  Interact with OpenAI-compatible APIs — the de facto standard for LLM inference used by
  vLLM, Ollama, LiteLLM, TGI, llama.cpp, and OpenAI itself. Use when: (1) Making chat
  completions or text completions requests, (2) Implementing tool/function calling,
  (3) Using structured outputs (JSON mode, json_schema), (4) Handling streaming responses
  (SSE), (5) Configuring sampling parameters (temperature, top_p, max_tokens, penalties),
  (6) Working with the Python SDK (sync/async clients, retries, timeouts), (7) Pointing
  clients at alternative backends (vLLM, Ollama, LiteLLM), (8) Handling errors and rate
  limiting, (9) Using embeddings, audio, or image endpoints.
---

# OpenAI API Spec

The OpenAI API is the standard interface for LLM inference. This skill covers the API specification — usable with **any** OpenAI-compatible backend (OpenAI, vLLM, Ollama, LiteLLM, TGI, llama.cpp server, etc.).

## Python SDK Setup

```bash
pip install openai
```

```python
from openai import OpenAI, AsyncOpenAI

# OpenAI (default)
client = OpenAI(api_key="sk-...")

# Alternative backends — just change base_url
client = OpenAI(base_url="http://localhost:8000/v1", api_key="na")      # vLLM
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama") # Ollama
client = OpenAI(base_url="http://localhost:4000/v1", api_key="sk-...")  # LiteLLM
client = OpenAI(base_url="http://localhost:8080/v1", api_key="na")      # llama.cpp

# Async client
aclient = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="na")

# Retry and timeout config
client = OpenAI(
    api_key="sk-...",
    max_retries=3,           # auto-retry on 429, 500, 503
    timeout=60.0,            # request timeout in seconds
)
```

## Chat Completions

The primary endpoint for conversational LLMs.

### Basic Request

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful ML engineer."},
        {"role": "user", "content": "Explain attention mechanisms."},
    ],
    temperature=0.7,
    max_tokens=1024,
)

print(response.choices[0].message.content)
print(f"Tokens: {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")
```

### Message Roles

| Role | Purpose |
|------|---------|
| `system` | Sets behavior, persona, instructions |
| `user` | Human input |
| `assistant` | Model response (include for multi-turn context) |
| `tool` | Result of a tool call (paired with `tool_call_id`) |

### Multi-Turn Conversation

```python
messages = [
    {"role": "system", "content": "You are an ML assistant."},
    {"role": "user", "content": "What is LoRA?"},
    {"role": "assistant", "content": "LoRA (Low-Rank Adaptation) is..."},
    {"role": "user", "content": "How does it compare to full fine-tuning?"},
]
response = client.chat.completions.create(model="gpt-4o", messages=messages)
```

### Sampling Parameters

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=0.7,          # 0.0 = deterministic, 2.0 = max randomness
    top_p=0.9,                # nucleus sampling (1.0 = disabled)
    max_tokens=2048,          # max output tokens
    stop=["\n\n", "END"],     # stop sequences
    frequency_penalty=0.5,    # reduce repetition of frequent tokens (-2.0 to 2.0)
    presence_penalty=0.3,     # encourage new topics (-2.0 to 2.0)
    seed=42,                  # deterministic sampling (best-effort)
    n=1,                      # number of completions
    logprobs=True,            # return log probabilities
    top_logprobs=5,           # number of top logprobs per token
)
```

**Parameter guidance:**
- **Factual/code tasks**: `temperature=0.0` or `temperature=0.1`
- **Creative tasks**: `temperature=0.7-1.0`
- **Don't mix** `temperature` and `top_p` — use one or the other
- **`max_tokens`** limits output only; prompt tokens are separate

## Streaming

### SSE (Server-Sent Events)

```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a haiku about GPUs."}],
    stream=True,
    stream_options={"include_usage": True},  # usage in final chunk
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
    if chunk.usage:
        print(f"\nTokens: {chunk.usage.total_tokens}")
```

### Async Streaming

```python
async def stream_response(messages):
    stream = await aclient.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```

### Raw SSE Format

```
data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## Tool Calling (Function Calling)

### Define Tools

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_gpu_metrics",
            "description": "Get current GPU utilization, memory usage, and temperature",
            "parameters": {
                "type": "object",
                "properties": {
                    "gpu_id": {
                        "type": "integer",
                        "description": "GPU device index (0-based)",
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["utilization", "memory", "temperature"]},
                        "description": "Which metrics to retrieve",
                    },
                },
                "required": ["gpu_id"],
                "additionalProperties": False,
            },
        },
    },
]
```

### Handle Tool Calls

```python
import json

messages = [{"role": "user", "content": "What's the GPU 0 utilization?"}]

# Step 1: Model decides to call a tool
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="auto",  # auto, none, required, or {"type":"function","function":{"name":"..."}}
)

msg = response.choices[0].message

if msg.tool_calls:
    # Step 2: Execute the tool
    messages.append(msg)  # include assistant message with tool_calls

    for tool_call in msg.tool_calls:
        args = json.loads(tool_call.function.arguments)
        result = get_gpu_metrics(**args)  # your function

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result),
        })

    # Step 3: Model generates final response with tool results
    final = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )
    print(final.choices[0].message.content)
```

### Parallel Tool Calls

The model may return multiple `tool_calls` in a single response. Process all of them and include all results before the next request:

```python
# msg.tool_calls = [tool_call_1, tool_call_2, ...]
for tool_call in msg.tool_calls:
    result = dispatch_tool(tool_call.function.name, tool_call.function.arguments)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result),
    })
```

Set `parallel_tool_calls=False` to force one tool call at a time.

## Structured Outputs

### JSON Mode

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Respond in JSON."},
        {"role": "user", "content": "List 3 GPU models with their VRAM."},
    ],
    response_format={"type": "json_object"},
)
# Guaranteed valid JSON (but no schema enforcement)
data = json.loads(response.choices[0].message.content)
```

### JSON Schema (Strict Mode)

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Describe the A100 GPU."}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "gpu_info",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "vram_gb": {"type": "integer"},
                    "architecture": {"type": "string"},
                    "use_cases": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["name", "vram_gb", "architecture", "use_cases"],
                "additionalProperties": False,
            },
        },
    },
)
```

**`strict: true` rules:**
- All fields must be `required`
- Must set `additionalProperties: false` at every object level
- Supports: string, number, integer, boolean, array, object, enum, anyOf (for nullable)
- For optional fields, use `anyOf: [{"type": "string"}, {"type": "null"}]`

### With Pydantic (SDK Helper)

```python
from pydantic import BaseModel

class GPUInfo(BaseModel):
    name: str
    vram_gb: int
    architecture: str
    use_cases: list[str]

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Describe the A100 GPU."}],
    response_format=GPUInfo,
)
gpu = response.choices[0].message.parsed  # typed GPUInfo object
```

## Text Completions (Legacy)

```python
response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt="The key innovation of transformers is",
    max_tokens=100,
    temperature=0.0,
)
print(response.choices[0].text)
```

> Most backends still support this endpoint. Useful for fill-in-the-middle or non-chat models.

## Embeddings

```python
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["What is PagedAttention?", "How does KV caching work?"],
    encoding_format="float",  # or "base64" for compact transfer
    dimensions=512,           # optional: truncate dimensions (some models)
)

for item in response.data:
    print(f"Index {item.index}: {len(item.embedding)} dims")
```

## Models Endpoint

```python
# List available models
models = client.models.list()
for m in models.data:
    print(m.id)

# Get specific model
model = client.models.retrieve("gpt-4o")
```

## Error Handling

```python
from openai import (
    APIError,
    APIConnectionError,
    RateLimitError,
    APIStatusError,
    AuthenticationError,
)

try:
    response = client.chat.completions.create(...)
except RateLimitError as e:
    # 429 — back off and retry (SDK auto-retries by default)
    print(f"Rate limited: {e}")
except AuthenticationError as e:
    # 401 — invalid API key
    print(f"Auth failed: {e}")
except APIConnectionError as e:
    # Network error — server unreachable
    print(f"Connection failed: {e}")
except APIStatusError as e:
    # Other HTTP errors (400, 500, etc.)
    print(f"API error {e.status_code}: {e.message}")
```

### Common Error Codes

| Code | Meaning | Fix |
|------|---------|-----|
| 400 | Bad request (invalid params) | Check message format, parameter values |
| 401 | Authentication failed | Check API key |
| 403 | Forbidden | Check permissions / model access |
| 404 | Model not found | Verify model name: `client.models.list()` |
| 429 | Rate limited | Back off; SDK retries automatically |
| 500 | Server error | Retry; if persistent, check backend logs |
| 503 | Server overloaded | Retry with backoff |

### Manual Retry with Backoff

```python
import time
from openai import RateLimitError

def call_with_retry(fn, max_retries=5):
    for attempt in range(max_retries):
        try:
            return fn()
        except RateLimitError:
            wait = 2 ** attempt
            print(f"Rate limited, waiting {wait}s...")
            time.sleep(wait)
    raise Exception("Max retries exceeded")
```

## Token Counting

```python
import tiktoken

# Get encoding for a model
enc = tiktoken.encoding_for_model("gpt-4o")

# Count tokens
text = "Hello, how are you?"
tokens = enc.encode(text)
print(f"{len(tokens)} tokens")

# Chat message token counting (approximate)
def count_chat_tokens(messages, model="gpt-4o"):
    enc = tiktoken.encoding_for_model(model)
    total = 0
    for msg in messages:
        total += 4  # overhead per message
        total += len(enc.encode(msg["content"]))
        total += len(enc.encode(msg["role"]))
    total += 2  # reply priming
    return total
```

**Context window management:**
- Track cumulative tokens across multi-turn conversations
- Truncate oldest messages (keep system prompt) when approaching the limit
- Use `max_tokens` to reserve space for the response

## Vision (Image Inputs)

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {
                "url": "https://example.com/image.jpg",
                "detail": "high",  # low | high | auto
            }},
        ],
    }],
    max_tokens=300,
)
```

Also supports base64: `{"url": f"data:image/png;base64,{b64_data}"}`

## Batch API

Process large request batches asynchronously (50% cheaper):

```python
import json

# 1. Create JSONL file
requests = [
    {"custom_id": f"req-{i}", "method": "POST", "url": "/v1/chat/completions",
     "body": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": f"Question {i}"}]}}
    for i in range(1000)
]
with open("batch_input.jsonl", "w") as f:
    for r in requests:
        f.write(json.dumps(r) + "\n")

# 2. Upload and create batch
batch_file = client.files.create(file=open("batch_input.jsonl", "rb"), purpose="batch")
batch = client.batches.create(input_file_id=batch_file.id, endpoint="/v1/chat/completions", completion_window="24h")

# 3. Poll status
status = client.batches.retrieve(batch.id)
# When status.status == "completed":
result_file = client.files.content(status.output_file_id)
```

## Audio (Whisper + TTS)

```python
# Speech-to-text
transcription = client.audio.transcriptions.create(
    model="whisper-1", file=open("audio.mp3", "rb"), language="en"
)

# Text-to-speech
response = client.audio.speech.create(model="tts-1", voice="alloy", input="Hello world")
response.stream_to_file("output.mp3")
```

## Backend Compatibility

| Feature | OpenAI | vLLM | Ollama | LiteLLM | llama.cpp |
|---------|--------|------|--------|---------|-----------|
| Chat completions | ✅ | ✅ | ✅ | ✅ | ✅ |
| Streaming | ✅ | ✅ | ✅ | ✅ | ✅ |
| Tool calling | ✅ | ✅ | ✅ | ✅ | ✅ |
| JSON mode | ✅ | ✅ | ✅ | ✅ | ✅ |
| JSON schema (strict) | ✅ | ✅ | Partial | Via proxy | Partial |
| Embeddings | ✅ | ✅ | ✅ | ✅ | ✅ |
| Completions (legacy) | ✅ | ✅ | ❌ | ✅ | ✅ |
| Logprobs | ✅ | ✅ | ❌ | Via proxy | ✅ |
| `seed` | ✅ | ✅ | ✅ | Via proxy | ✅ |

## Reference

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [openai-python SDK](https://github.com/openai/openai-python)
- [tiktoken](https://github.com/openai/tiktoken)
- `references/patterns.md` — common integration patterns
