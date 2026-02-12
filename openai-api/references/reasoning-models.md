# Reasoning / Thinking Models

GPT-5, GPT-5-mini, GPT-5-nano, o3, o4-mini, and successors use extended reasoning (chain-of-thought) before responding. The **Responses API** is the recommended interface for reasoning models.

## Responses API (Recommended)

```python
response = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "medium"},
    input=[
        {"role": "user", "content": "Prove that sqrt(2) is irrational."},
    ],
)
print(response.output_text)

# Access reasoning token usage
print(response.usage.output_tokens_details.reasoning_tokens)
```

## Chat Completions API (Also Supported)

```python
response = client.chat.completions.create(
    model="gpt-5",
    messages=[{"role": "user", "content": "Prove that sqrt(2) is irrational."}],
    reasoning_effort="medium",
    max_completion_tokens=25000,       # includes reasoning + visible output
)
print(response.choices[0].message.content)
```

## Reasoning Effort

| Level | Behavior | Use Case |
|---|---|---|
| `none` | Skip reasoning entirely (GPT-5.1+) | Fast, no-reasoning tasks |
| `low` | Minimal reasoning, faster | Simple tasks, classification |
| `medium` | Balanced (default) | General use |
| `high` | Extended reasoning, slower | Math proofs, complex code, hard problems |

## Key Differences from Standard Chat Models

- **`developer` role** instead of `system`:
  ```python
  # Responses API
  input=[
      {"role": "developer", "content": "You are a math tutor."},
      {"role": "user", "content": "Solve this integral..."},
  ]

  # Chat Completions API
  messages=[
      {"role": "developer", "content": "You are a math tutor."},
      {"role": "user", "content": "Solve this integral..."},
  ]
  ```
- **`max_output_tokens`** (Responses) or **`max_completion_tokens`** (Chat) — includes both reasoning and visible tokens. Reserve at least 25,000 tokens when experimenting.
- **Temperature/top_p not supported** — reasoning models control their own sampling
- **Reasoning tokens are hidden** — they don't appear in output but are billed as output tokens

## Handling Incomplete Responses

If reasoning exhausts `max_output_tokens`, you may get no visible output:

```python
response = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "high"},
    input=[{"role": "user", "content": prompt}],
    max_output_tokens=5000,
)

if response.status == "incomplete" and response.incomplete_details.reason == "max_output_tokens":
    if response.output_text:
        print("Partial output:", response.output_text)
    else:
        print("Ran out of tokens during reasoning — increase max_output_tokens")
```

## Reasoning Summaries

Get concise summaries of the model's internal reasoning process:

```python
response = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "high", "summary": "auto"},  # or "concise"
    input=[{"role": "user", "content": "Design a distributed training pipeline."}],
)

# Access reasoning summaries in output items
for item in response.output:
    if item.type == "reasoning":
        for summary in item.summary:
            print(f"Reasoning: {summary.text}")
```

## Encrypted Reasoning Items (Stateless Mode)

When `store=False` or under zero data retention, pass reasoning items across turns:

```python
response = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "medium"},
    input=[{"role": "user", "content": "Step 1..."}],
    include=["reasoning.encrypted_content"],
    store=False,
)

# Pass reasoning items back in next turn
next_response = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "medium"},
    input=[
        *response.output,  # includes encrypted reasoning items
        {"role": "user", "content": "Now step 2..."},
    ],
    include=["reasoning.encrypted_content"],
    store=False,
)
```

## Reasoning with Tool Calling

Pass reasoning items back when doing multi-turn function calling:

```python
response = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "medium"},
    input=[{"role": "user", "content": "What's the weather in SF and NYC?"}],
    tools=tools,
)

# Execute tool calls, then pass back ALL output items (including reasoning)
next_response = client.responses.create(
    model="gpt-5",
    input=[
        {"role": "user", "content": "What's the weather in SF and NYC?"},
        *response.output,        # reasoning + function_call items
        {"type": "function_call_output", "call_id": call_id, "output": result},
    ],
    tools=tools,
)
```

## Model Selection

| Model | Speed | Reasoning | Cost | Best For |
|---|---|---|---|---|
| `gpt-5-nano` | Fastest | Basic | Lowest | Simple tasks, high volume |
| `gpt-5-mini` | Fast | Good | Low | Balanced workloads |
| `gpt-5` | Medium | Strong | Medium | Complex reasoning, broad domains |
| `o3` | Slow | Strongest | High | Research, hard math/code |
| `o4-mini` | Fast | Strong | Low | Efficient reasoning, 128K context |

## Backend Compatibility

| Feature | OpenAI | vLLM | Ollama | LiteLLM |
|---|---|---|---|---|
| Responses API | ✅ | ❌ | ❌ | Via proxy |
| Reasoning models | ✅ (GPT-5, o3, o4-mini) | ❌ | Partial (DeepSeek-R1) | Via proxy |
| `reasoning.effort` | ✅ | ❌ | ❌ | Via proxy |
| `developer` role | ✅ | ❌ | ❌ | Via proxy |

**Open-source reasoning:** DeepSeek-R1 and QwQ expose thinking in `<think>` tags within response content. They use standard chat completions (not the Responses API), so they work with any backend. GPT-OSS models from OpenAI also support the `developer` role via their chat template.
