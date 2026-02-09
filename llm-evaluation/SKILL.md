---
name: llm-evaluation
description: >
  Evaluate LLMs with lm-evaluation-harness and standard benchmarks. Use when: (1) Running
  benchmarks (MMLU, HumanEval, GSM8K, HellaSwag, TruthfulQA, etc.), (2) Configuring model
  backends and model_args, (3) Evaluating via HuggingFace, vLLM, or OpenAI-compatible endpoints,
  (4) Writing custom evaluation tasks and task groups, (5) Configuring few-shot, chat templates,
  filters, and generation settings, (6) Understanding metrics (exact_match, acc, pass@k),
  (7) Building evaluation pipelines for model selection, (8) LLM-as-judge evaluation.
---

# LLM Evaluation

[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (EleutherAI) — the standard framework with 60+ benchmarks. Version: **0.4.x+**.

## Model Backends and Configuration

### Backend Reference

| Backend | `model` value | When to Use |
|---|---|---|
| HuggingFace (local) | `hf` | Direct model loading on GPU |
| vLLM (local) | `vllm` | Fast GPU inference, tensor parallel |
| OpenAI-compatible (completions) | `local-completions` | Existing vLLM/Ollama server |
| OpenAI-compatible (chat) | `local-chat-completions` | Chat-format endpoints |
| OpenAI API | `openai-completions` | OpenAI hosted models |

### model_args Reference

All backends accept `model_args` as a comma-separated `key=value` string.

**HuggingFace (`hf`) model_args:**

| Arg | Purpose | Default |
|---|---|---|
| `pretrained` | Model ID or local path | required |
| `dtype` | Weight dtype (`auto`, `float16`, `bfloat16`) | `auto` |
| `revision` | Model revision/branch | `main` |
| `trust_remote_code` | Allow custom model code | `False` |
| `parallelize` | Naive model parallelism across GPUs | `False` |
| `max_length` | Override max context length | Model default |
| `device` | Device (`cuda`, `cuda:0`, `cpu`) | Auto |
| `peft` | Path to PEFT/LoRA adapter | None |
| `delta` | Path to delta weights | None |
| `autogptq` | Use AutoGPTQ quantization | `False` |
| `add_bos_token` | Prepend BOS token | `False` |

**vLLM (`vllm`) model_args:**

| Arg | Purpose | Default |
|---|---|---|
| `pretrained` | Model ID or local path | required |
| `dtype` | Weight dtype | `auto` |
| `tensor_parallel_size` | TP across GPUs | `1` |
| `gpu_memory_utilization` | KV cache memory fraction | `0.9` |
| `max_model_len` | Max context length | Model default |
| `quantization` | Quantization method | None |
| `trust_remote_code` | Allow custom code | `False` |
| `data_parallel_size` | Data parallel replicas | `1` |
| `max_num_seqs` | Max concurrent sequences | `256` |

**API (`local-completions` / `local-chat-completions`) model_args:**

| Arg | Purpose | Default |
|---|---|---|
| `model` | Model name in API | required |
| `base_url` | Server base URL | required |
| `tokenizer_backend` | Tokenizer source (`huggingface`) | None |
| `num_concurrent` | Concurrent API requests | `1` |
| `max_retries` | Retry count | `3` |
| `tokenized_requests` | Send token IDs instead of text | `False` |

### Python API

```python
import lm_eval

results = lm_eval.simple_evaluate(
    model="vllm",
    model_args="pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=auto,tensor_parallel_size=2,gpu_memory_utilization=0.9",
    tasks=["mmlu", "gsm8k", "hellaswag"],
    num_fewshot=5,
    batch_size="auto",
    log_samples=True,
    apply_chat_template=True,
)

for task_name, task_result in results["results"].items():
    print(f"{task_name}: {task_result}")
```

## Evaluation Settings

### Core Settings

| Setting | Purpose | Values |
|---|---|---|
| `tasks` | Benchmark tasks to run | Comma-separated names or list |
| `num_fewshot` | Few-shot examples | Integer (0 = zero-shot) |
| `batch_size` | Batch size | Integer or `"auto"` |
| `limit` | Sample limit per task | Integer (count) or float (fraction) |
| `apply_chat_template` | Use model's chat template | `True`/`False` |
| `fewshot_as_multiturn` | Few-shot as multi-turn conversation | `True`/`False` |
| `system_instruction` | System prompt for chat template | String |
| `log_samples` | Log individual predictions | `True`/`False` |
| `output_path` | Results output directory | Path string |
| `seed` | Random seed for reproducibility | List `[0,1234,1234,1234]` (default) |
| `use_cache` | Cache model responses | Path string or None |

### Generation Settings (for generate_until tasks)

These are set per-task in YAML configs, not globally:

```yaml
generation_kwargs:
  max_gen_toks: 1024          # max tokens to generate
  temperature: 0.0            # 0.0 = greedy
  top_p: 1.0
  do_sample: false
  stop_sequences: ["\n\n"]    # stop on these strings
  until: ["\n\nQuestion:"]    # alternative stop format
```

### Filter Pipeline

Filters post-process model output before metric computation:

```yaml
filter_list:
  - name: get-answer
    filter:
      - function: regex
        regex_pattern: "#### (\\d+)"   # extract final answer
      - function: take_first
  - name: remove-whitespace
    filter:
      - function: strip
      - function: lowercase
```

Built-in filter functions: `regex`, `take_first`, `strip`, `lowercase`, `uppercase`, `remove_whitespace`, `map`, `at`.

### W&B Integration

```python
results = lm_eval.simple_evaluate(
    model="vllm",
    model_args="pretrained=my-model",
    tasks=["mmlu"],
    wandb_args="project=eval,name=llama-8b-mmlu,job_type=eval",
)
```

## Benchmarks

### Knowledge and Reasoning

| Benchmark | Task | Metric | Shots | What It Measures |
|-----------|------|--------|-------|-----------------|
| **MMLU** | `mmlu` | acc | 5 | Broad knowledge (57 subjects) |
| **MMLU-Pro** | `mmlu_pro` | acc | 5 | Harder MMLU (10 answer choices, CoT) |
| **ARC-Challenge** | `arc_challenge` | acc_norm | 25 | Grade-school science reasoning |
| **HellaSwag** | `hellaswag` | acc_norm | 10 | Commonsense completion |
| **Winogrande** | `winogrande` | acc | 5 | Commonsense coreference |
| **TruthfulQA** | `truthfulqa_mc2` | mc2 | 0 | Resistance to misconceptions |
| **GPQA** | `gpqa_main_zeroshot` | acc | 0 | Graduate-level QA |
| **BBH** | `bbh` | acc | 3 | Big Bench Hard (23 tasks, CoT) |
| **MuSR** | `musr` | acc | 0 | Multi-step reasoning |

### Math and Code

| Benchmark | Task | Metric | Shots | What It Measures |
|-----------|------|--------|-------|-----------------|
| **GSM8K** | `gsm8k` | exact_match | 5 | Grade-school math (multi-step) |
| **MATH** | `minerva_math` | exact_match | 4 | Competition mathematics |
| **MATH-Hard** | `minerva_math_hard` | exact_match | 4 | Hard subset of MATH |
| **HumanEval** | `humaneval` | pass@1 | 0 | Python code generation |
| **MBPP** | `mbpp` | pass@1 | 3 | Python programming |

### Instruction Following

| Benchmark | Task | Metric | Shots | What It Measures |
|-----------|------|--------|-------|-----------------|
| **IFEval** | `ifeval` | strict_acc | 0 | Verifiable instruction following |

### Open LLM Leaderboard v2

```python
results = lm_eval.simple_evaluate(
    model="vllm",
    model_args="pretrained=my-model,tensor_parallel_size=4",
    tasks=["mmlu_pro", "gpqa_main_zeroshot", "musr", "minerva_math_hard", "ifeval", "bbh"],
    batch_size="auto",
    apply_chat_template=True,
)
```

## Metrics Reference

| Metric | Description | Used By |
|--------|-------------|---------|
| `acc` | Accuracy (multiple choice) | MMLU, ARC, BoolQ |
| `acc_norm` | Length-normalized accuracy | HellaSwag, ARC, PIQA |
| `exact_match` | Exact string match (after normalization) | GSM8K, MATH |
| `exact_match,strict-match` | Strict exact match | GSM8K |
| `exact_match,flexible-extract` | Flexible number extraction | GSM8K |
| `pass@k` | k/n code samples pass tests | HumanEval, MBPP |
| `f1` | Token-level F1 | SQuAD |
| `mc2` | Weighted multi-choice accuracy | TruthfulQA |
| `word_perplexity` | Word-level perplexity | WikiText |
| `byte_perplexity` | Byte-level perplexity | WikiText |

### Metric Gotchas

- **`acc` vs `acc_norm`** — HellaSwag scores differ by 5+ points depending on which you use; always use `acc_norm`
- **GSM8K exact_match** — answer extraction regex matters hugely; `strict-match` vs `flexible-extract` can differ by 10+ points
- **Few-shot count** — MMLU 0-shot vs 5-shot can differ by 10+ points; always report shot count
- **Chat template** — instruction-tuned models need `apply_chat_template=True` or scores will be significantly lower

## Custom Evaluation Tasks

### Task YAML Schema

```yaml
task: my_custom_task
dataset_path: json                    # or huggingface dataset name
dataset_name: null
dataset_kwargs:
  data_files:
    test: ./data/test.jsonl
output_type: generate_until           # or multiple_choice, loglikelihood
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
filter_list:
  - name: get-answer
    filter:
      - function: regex
        regex_pattern: "(\\d+)"
      - function: take_first
num_fewshot: 0
```

### output_type Reference

| Type | Description | Metrics |
|---|---|---|
| `multiple_choice` | Model scores each option | `acc`, `acc_norm` |
| `loglikelihood` | Log probability of target | `perplexity`, `acc` |
| `loglikelihood_rolling` | Rolling log probability | `word_perplexity`, `byte_perplexity` |
| `generate_until` | Free-form generation | `exact_match`, `pass@k`, custom |

### Task Groups

```yaml
group: my_eval_suite
task:
  - my_custom_task
  - mmlu
  - gsm8k
aggregate_metric_list:
  - metric: acc
    aggregation: mean
    weight_by_size: true
```

### Task Inheritance

```yaml
include: _default_template.yaml
task: my_variant
dataset_name: hard_subset
num_fewshot: 0
```

## LLM-as-Judge Evaluation

```python
import openai

def judge_output(prompt, response, reference):
    judge_prompt = f"""Rate the following response on a scale of 1-5.
Question: {prompt}
Reference Answer: {reference}
Model Response: {response}
Score (1-5):"""

    result = openai.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.0,
    )
    return int(result.choices[0].message.content.strip())
```

For structured LLM-as-judge, see [MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) and [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval).

## Best Practices

1. **Consistent settings** across compared models — same shots, batch size, chat template
2. **Report standard error** — `_stderr` fields in results; small differences may be noise
3. **Multiple tasks** — no single benchmark tells the whole story
4. **vLLM backend** for faster GPU evaluation (`tensor_parallel_size` for large models)
5. **Save raw results** (`log_samples=True`) for debugging unexpected scores
6. **Match eval to use case** — coding? HumanEval. Reasoning? GSM8K/MATH. General? MMLU.

## Cross-References

- [vllm](../vllm/) — vLLM backend for fast GPU evaluation
- [openai-api](../openai-api/) — OpenAI-compatible API backends (local-completions, local-chat-completions)
- [wandb](../wandb/) — Log evaluation results to W&B

## Reference

- [lm-evaluation-harness GitHub](https://github.com/EleutherAI/lm-evaluation-harness)
- [Available tasks](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks)
- [Task YAML guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md)
- [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
- `references/benchmarks.md` — detailed benchmark descriptions and scoring baselines
