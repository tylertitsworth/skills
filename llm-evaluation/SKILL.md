---
name: llm-evaluation
description: >
  Evaluate LLMs with lm-evaluation-harness and standard benchmarks. Use when: (1) Running
  benchmarks (MMLU, HumanEval, GSM8K, HellaSwag, TruthfulQA, etc.), (2) Using lm-eval CLI
  or Python API, (3) Evaluating models via HuggingFace, vLLM, or OpenAI-compatible endpoints,
  (4) Writing custom evaluation tasks, (5) Comparing models across benchmarks, (6) Configuring
  few-shot evaluation, (7) Understanding metrics (exact_match, acc, pass@k, perplexity),
  (8) Building evaluation pipelines for model selection.
---

# LLM Evaluation

Evaluate language models using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (EleutherAI) — the standard framework with 60+ benchmarks. Version: **0.4.x+**.

## Setup

```bash
# Add to container image: lm-eval
# With vLLM backend
# With vLLM backend: lm-eval[vllm]
```

## Quick Start

```bash
# Evaluate a HuggingFace model on MMLU (5-shot)
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
  --tasks mmlu \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path ./results

# Multiple tasks
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=bfloat16 \
  --tasks mmlu,gsm8k,hellaswag,truthfulqa_mc2 \
  --batch_size auto \
  --output_path ./results
```

## Model Backends

### HuggingFace (Local)

```bash
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=bfloat16,trust_remote_code=True \
  --tasks mmlu --batch_size auto
```

### vLLM (Fast GPU Inference)

```bash
lm_eval --model vllm \
  --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=auto,tensor_parallel_size=2,gpu_memory_utilization=0.8 \
  --tasks gsm8k --batch_size auto
```

### OpenAI-Compatible API (vLLM server, Ollama, etc.)

```bash
lm_eval --model local-completions \
  --model_args model=meta-llama/Llama-3.1-8B-Instruct,base_url=http://localhost:8000/v1,tokenizer_backend=huggingface \
  --tasks mmlu --batch_size 16

# Chat completions endpoint
lm_eval --model local-chat-completions \
  --model_args model=meta-llama/Llama-3.1-8B-Instruct,base_url=http://localhost:8000/v1 \
  --tasks mmlu --batch_size 16 --apply_chat_template
```

### OpenAI API

```bash
# Set OPENAI_API_KEY env var in pod spec from a K8s Secret
lm_eval --model openai-completions \
  --model_args model=gpt-4o \
  --tasks mmlu --batch_size 16
```

## Common Benchmarks

### Knowledge and Reasoning

| Benchmark | Task | Metric | Shots | What It Measures |
|-----------|------|--------|-------|-----------------|
| **MMLU** | `mmlu` | acc | 5 | Broad knowledge (57 subjects) |
| **MMLU-Pro** | `mmlu_pro` | acc | 5 | Harder MMLU (10 answer choices) |
| **ARC-Challenge** | `arc_challenge` | acc_norm | 25 | Grade-school science reasoning |
| **HellaSwag** | `hellaswag` | acc_norm | 10 | Commonsense completion |
| **Winogrande** | `winogrande` | acc | 5 | Commonsense coreference |
| **TruthfulQA** | `truthfulqa_mc2` | acc | 0 | Resistance to misconceptions |

### Math and Code

| Benchmark | Task | Metric | Shots | What It Measures |
|-----------|------|--------|-------|-----------------|
| **GSM8K** | `gsm8k` | exact_match | 5 | Grade-school math (multi-step) |
| **MATH** | `minerva_math` | exact_match | 4 | Competition mathematics |
| **HumanEval** | `humaneval` | pass@1 | 0 | Python code generation |
| **MBPP** | `mbpp` | pass@1 | 3 | Python programming |

### Aggregate Suites

```bash
# Open LLM Leaderboard v2 tasks
lm_eval --model vllm \
  --model_args pretrained=my-model \
  --tasks mmlu_pro,gpqa_main_zeroshot,musr,minerva_math_hard,ifeval,bbh \
  --batch_size auto

# Common general evaluation
lm_eval --model vllm \
  --model_args pretrained=my-model \
  --tasks mmlu,gsm8k,hellaswag,arc_challenge,truthfulqa_mc2,winogrande \
  --batch_size auto
```

## Configuration Options

```bash
# Few-shot
--num_fewshot 5

# Batch size
--batch_size auto      # auto-detect optimal
--batch_size 32        # fixed

# Limit samples (quick testing)
--limit 100            # 100 samples per task
--limit 0.1            # 10% of samples

# Chat template
--apply_chat_template
--fewshot_as_multiturn  # few-shot as multi-turn conversation

# Output
--output_path ./results
--log_samples                  # log individual predictions
--wandb_args project=eval,name=llama-8b  # W&B logging
```

## Python API

```python
import lm_eval

results = lm_eval.simple_evaluate(
    model="hf",
    model_args="pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=bfloat16",
    tasks=["mmlu", "gsm8k", "hellaswag"],
    num_fewshot=5,
    batch_size="auto",
    log_samples=True,
)

for task_name, task_result in results["results"].items():
    print(f"{task_name}: {task_result}")
```

## Custom Evaluation Tasks

Create a YAML task config:

```yaml
# my_tasks/custom_qa.yaml
task: custom_qa
dataset_path: json
dataset_name: null
dataset_kwargs:
  data_files:
    test: ./data/custom_qa.jsonl
output_type: generate_until
doc_to_text: "Question: {{question}}\nAnswer:"
doc_to_target: "{{answer}}"
generation_kwargs:
  max_gen_toks: 256
  temperature: 0.0
  do_sample: false
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
num_fewshot: 0
```

**Data format (`custom_qa.jsonl`):**
```json
{"question": "What GPU architecture introduced tensor cores?", "answer": "Volta"}
{"question": "What does FSDP stand for?", "answer": "Fully Sharded Data Parallel"}
```

**Run:**
```bash
lm_eval --model hf --model_args pretrained=my-model \
  --tasks custom_qa --include_path ./my_tasks --batch_size auto
```

### Task Groups

```yaml
# my_tasks/_group.yaml
group: my_eval_suite
task:
  - custom_qa
  - mmlu
  - gsm8k
```

```bash
lm_eval --tasks my_eval_suite --include_path ./my_tasks
```

## Model Comparison

```bash
# Evaluate multiple models
for model in meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct; do
  lm_eval --model vllm \
    --model_args pretrained=$model,dtype=auto \
    --tasks mmlu,gsm8k,hellaswag,arc_challenge \
    --batch_size auto \
    --output_path ./results/$(basename $model)
done
```

```python
# Compare results
import json, pandas as pd
from pathlib import Path

models = {}
for d in Path("./results").iterdir():
    with open(d / "results.json") as f:
        data = json.load(f)
    models[d.name] = {
        task: metrics.get("acc,none", metrics.get("exact_match,strict-match", "N/A"))
        for task, metrics in data["results"].items()
    }
print(pd.DataFrame(models).T.to_markdown())
```

## Metrics Reference

| Metric | Description | Used By |
|--------|-------------|---------|
| `acc` | Accuracy (multiple choice) | MMLU, ARC, BoolQ |
| `acc_norm` | Length-normalized accuracy | HellaSwag, ARC, PIQA |
| `exact_match` | Exact string match | GSM8K, MATH |
| `pass@k` | Passes k of n code samples | HumanEval, MBPP |
| `f1` | Token-level F1 | SQuAD |
| `mc2` | Weighted multi-choice accuracy | TruthfulQA |

## Generation-Based Evaluation

For open-ended evaluation (not multiple choice):

```yaml
# generate_until task
output_type: generate_until
generation_kwargs:
  max_gen_toks: 1024
  temperature: 0.0
  stop_sequences: ["\n\nQuestion:", "---"]
filter_list:
  - name: get-answer
    filter:
      - function: regex
        regex_pattern: "#### (\\d+)"
      - function: take_first
```

## LLM-as-Judge Evaluation

Use a strong model to judge outputs:

```python
# Simple LLM-as-judge pattern
import openai

def judge_output(prompt, response, reference):
    judge_prompt = f"""Rate the following response on a scale of 1-5.
Question: {prompt}
Reference Answer: {reference}
Model Response: {response}
Score (1-5):"""

    result = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.0,
    )
    return int(result.choices[0].message.content.strip())
```

For structured LLM-as-judge, see [MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) and [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval).

## Alternative Evaluation Tools

| Tool | Strength | Use Case |
|---|---|---|
| **lm-evaluation-harness** | Standard benchmarks, many backends | Model comparison, leaderboard reproduction |
| **lighteval** | HuggingFace-native, fast | Quick evals, custom tasks |
| **HELM** | Comprehensive, standardized | Academic evaluation |
| **Inspect AI** | Agent and tool-use evaluation | Complex agentic tasks |
| **bigcode-eval-harness** | Code generation | HumanEval, MBPP, MultiPL-E |

### lighteval

```bash
# Add to container image: lighteval
lighteval accelerate --model_args pretrained=my-model \
  --tasks "leaderboard|mmlu:5|0,leaderboard|gsm8k|0" \
  --output_dir ./results
```

## Best Practices

1. **Consistent settings** across compared models (same shots, batch size)
2. **Report standard error** (`_stderr` fields) — small differences may be noise
3. **Multiple tasks** — no single benchmark tells the whole story
4. **vLLM backend** for faster GPU evaluation
5. **Save raw results** (`--log_samples`) for debugging unexpected scores
6. **Match eval to use case** — coding? HumanEval. Reasoning? GSM8K/MATH. General? MMLU.

## Reference

- [lm-evaluation-harness GitHub](https://github.com/EleutherAI/lm-evaluation-harness)
- [Available tasks](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks)
- [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
- `references/benchmarks.md` — detailed benchmark descriptions and scoring baselines
