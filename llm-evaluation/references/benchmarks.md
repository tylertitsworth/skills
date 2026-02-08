# LLM Benchmark Reference

## Benchmark Details

### MMLU (Massive Multitask Language Understanding)

- **Task**: `mmlu` | **Format**: 4-choice MC | **Size**: ~14,000 questions across 57 subjects
- **Default shots**: 5 | **Metric**: `acc`
- **Baselines**: Random = 25%, Llama-3.1-8B ~65%, GPT-4 ~86%

```bash
lm_eval --tasks mmlu --num_fewshot 5
# Specific subjects:
lm_eval --tasks mmlu_abstract_algebra,mmlu_computer_security
```

### GSM8K (Grade School Math 8K)

- **Task**: `gsm8k` | **Format**: Open-ended (chain-of-thought extraction)
- **Size**: 1,319 test problems | **Default shots**: 5
- **Metric**: `exact_match` (strict-match and flexible-extract)
- **Baselines**: Llama-3.1-8B ~56%, GPT-4 ~95%

```bash
lm_eval --tasks gsm8k --num_fewshot 5
# With chain-of-thought:
lm_eval --tasks gsm8k_cot --num_fewshot 8
```

### HumanEval

- **Task**: `humaneval` | **Format**: Python code generation with test cases
- **Size**: 164 problems | **Metric**: `pass@1`
- **Baselines**: Llama-3.1-8B ~33%, GPT-4 ~87%
- **Note**: Requires code execution — use sandboxed environments

### HellaSwag

- **Task**: `hellaswag` | **Format**: 4-choice MC (commonsense completion)
- **Size**: 10,042 test | **Default shots**: 10 | **Metric**: `acc_norm`
- **Baselines**: Llama-3.1-8B ~79%, GPT-4 ~95%

### TruthfulQA

- **Task**: `truthfulqa_mc2` | **Format**: Multiple choice (weighted multi-correct)
- **Size**: 817 questions | **Default shots**: 0 | **Metric**: `acc`
- **Note**: MC2 (weighted) is preferred over MC1 (single correct)

### ARC (AI2 Reasoning Challenge)

- **Task**: `arc_challenge` (hard) / `arc_easy`
- **Format**: 4-choice MC | **Default shots**: 25 | **Metric**: `acc_norm`

### Winogrande

- **Task**: `winogrande` | **Format**: Binary choice (fill-in-the-blank)
- **Default shots**: 5 | **Metric**: `acc`

## Open LLM Leaderboard v2

| Benchmark | Task | Shots | Metric |
|-----------|------|-------|--------|
| MMLU-Pro | `mmlu_pro` | 5 | acc |
| GPQA | `gpqa_main_zeroshot` | 0 | acc_norm |
| MuSR | `musr` | 0 | acc_norm |
| MATH (Hard) | `minerva_math_hard` | 4 | exact_match |
| IFEval | `ifeval` | 0 | inst_level_strict_acc |
| BBH | `bbh` | 3 | acc_norm |

## Choosing Benchmarks by Use Case

| Use Case | Recommended |
|----------|-------------|
| General capability | MMLU, HellaSwag, ARC, TruthfulQA |
| Math/reasoning | GSM8K, MATH, ARC-Challenge |
| Coding | HumanEval, MBPP |
| Instruction following | IFEval, MT-Bench (separate tool) |
| Safety | TruthfulQA, BBQ |
| Domain-specific | Custom tasks (see SKILL.md) |

## Interpreting Results

### Standard Error

Always check `_stderr` — a 0.5% difference with ±0.4% stderr is not significant:
```
mmlu: acc = 0.654 ± 0.004
```

### Contamination

Models trained on benchmark data show inflated scores. Red flags:
- Suspiciously high on specific benchmarks vs peers
- Much higher than similar-sized models

### Normalized vs Raw Accuracy

`acc_norm` (length-normalized) is fairer for MC — prevents favoring shorter answers. Use `acc_norm` when available.
