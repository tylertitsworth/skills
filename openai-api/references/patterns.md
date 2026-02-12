# OpenAI API Common Patterns

## Multi-Turn Tool Use Loop

A complete pattern for agentic tool use — loop until the model stops calling tools:

```python
import json
from openai import OpenAI

def run_agent(client: OpenAI, model: str, messages: list, tools: list, max_turns: int = 10):
    """Run an agent loop that handles tool calls until completion."""
    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content  # done — no more tool calls

        # Execute all tool calls
        for tool_call in msg.tool_calls:
            result = execute_tool(tool_call.function.name,
                                  json.loads(tool_call.function.arguments))
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

    return "Max turns reached"

def execute_tool(name: str, args: dict) -> dict:
    """Dispatch tool calls to actual implementations."""
    tools_registry = {
        "get_gpu_metrics": get_gpu_metrics,
        "run_benchmark": run_benchmark,
    }
    fn = tools_registry.get(name)
    if fn:
        return fn(**args)
    return {"error": f"Unknown tool: {name}"}
```

## Streaming with Tool Calls

Tool calls arrive as deltas during streaming — accumulate them:

```python
def stream_with_tools(client, model, messages, tools):
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        stream=True,
    )

    content = ""
    tool_calls = {}  # index -> {id, name, arguments}

    for chunk in stream:
        delta = chunk.choices[0].delta

        # Accumulate content
        if delta.content:
            content += delta.content
            print(delta.content, end="", flush=True)

        # Accumulate tool calls
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls:
                    tool_calls[idx] = {"id": tc.id, "name": tc.function.name, "arguments": ""}
                if tc.function.arguments:
                    tool_calls[idx]["arguments"] += tc.function.arguments

    finish_reason = chunk.choices[0].finish_reason

    if finish_reason == "tool_calls":
        return list(tool_calls.values())
    return content
```

## Batch Requests (Parallel)

Process many requests concurrently with asyncio:

```python
import asyncio
from openai import AsyncOpenAI

async def batch_completions(prompts: list[str], model: str = "gpt-4o", max_concurrent: int = 10):
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_one(prompt):
        async with semaphore:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response.choices[0].message.content

    tasks = [process_one(p) for p in prompts]
    return await asyncio.gather(*tasks)

# Usage
results = asyncio.run(batch_completions([
    "Summarize transformer architecture.",
    "Explain gradient descent.",
    "What is backpropagation?",
]))
```

## Context Window Management

Truncate conversation history to stay within context limits:

```python
import tiktoken

def truncate_messages(
    messages: list[dict],
    max_tokens: int = 8000,
    model: str = "gpt-4o",
    reserve_for_response: int = 1024,
) -> list[dict]:
    """Keep system prompt + most recent messages that fit."""
    enc = tiktoken.encoding_for_model(model)
    budget = max_tokens - reserve_for_response

    # Always keep system messages
    system_msgs = [m for m in messages if m["role"] == "system"]
    other_msgs = [m for m in messages if m["role"] != "system"]

    # Count system token cost
    system_tokens = sum(len(enc.encode(m["content"])) + 4 for m in system_msgs)
    remaining = budget - system_tokens

    # Add messages from most recent, working backwards
    kept = []
    for msg in reversed(other_msgs):
        msg_tokens = len(enc.encode(msg.get("content", "") or "")) + 4
        if remaining - msg_tokens < 0:
            break
        kept.insert(0, msg)
        remaining -= msg_tokens

    return system_msgs + kept
```

## Structured Output with Retry

Parse structured output with validation and retry:

```python
from pydantic import BaseModel, ValidationError

class AnalysisResult(BaseModel):
    summary: str
    confidence: float
    tags: list[str]

def get_structured(client, prompt: str, retries: int = 3) -> AnalysisResult:
    for attempt in range(retries):
        try:
            response = client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format=AnalysisResult,
            )
            result = response.choices[0].message.parsed
            if result:
                return result
        except ValidationError as e:
            if attempt == retries - 1:
                raise
            print(f"Validation failed (attempt {attempt + 1}): {e}")

    raise ValueError("Failed to get structured output")
```

## Multi-Backend Fallback

Try multiple backends, falling back on failure:

```python
from openai import OpenAI, APIError, APIConnectionError

backends = [
    {"base_url": "http://vllm:8000/v1", "api_key": "na", "model": "llama-3.1-8b"},
    {"base_url": "http://ollama:11434/v1", "api_key": "ollama", "model": "llama3.1"},
    {"base_url": "https://api.openai.com/v1", "api_key": "sk-...", "model": "gpt-4o-mini"},
]

def chat_with_fallback(messages: list[dict]) -> str:
    for backend in backends:
        try:
            client = OpenAI(base_url=backend["base_url"], api_key=backend["api_key"])
            response = client.chat.completions.create(
                model=backend["model"],
                messages=messages,
                timeout=30,
            )
            return response.choices[0].message.content
        except (APIError, APIConnectionError) as e:
            print(f"Backend {backend['base_url']} failed: {e}")
            continue
    raise Exception("All backends failed")
```

## Embeddings for Similarity Search

```python
import numpy as np

def get_embeddings(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Semantic search
query_emb = get_embeddings(["How does Flash Attention work?"])[0]
doc_embs = get_embeddings(documents)
similarities = [cosine_similarity(query_emb, d) for d in doc_embs]
top_idx = np.argmax(similarities)
print(f"Most similar: {documents[top_idx]}")
```

## cURL Examples

For debugging or quick testing:

```bash
# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token-abc123" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 100
  }'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token-abc123" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'

# List models
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer token-abc123"
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
