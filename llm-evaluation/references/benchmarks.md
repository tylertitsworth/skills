# LLM Benchmark Reference

Detailed descriptions, scoring baselines, and gotchas for common LLM benchmarks.

## Knowledge Benchmarks

### MMLU (Massive Multitask Language Understanding)

- **Tasks:** 57 subjects (STEM, humanities, social science, other)
- **Format:** 4-choice multiple choice
- **Standard:** 5-shot
- **Metric:** accuracy
- **Baselines:** Random = 25%, GPT-4o ≈ 88%, Llama 3.1 70B ≈ 83%
- **Gotchas:** Formatting sensitive — ensure answer extraction parses "A", "B", "C", "D" correctly. Some tasks are noisy or mislabeled.

### MMLU-Pro

- **Tasks:** MMLU subset with harder questions, 10 answer choices
- **Standard:** 5-shot, CoT (chain of thought) prompted
- **Metric:** accuracy
- **Baselines:** GPT-4o ≈ 73%, random = 10%
- **Gotchas:** Much harder than MMLU. Requires CoT for good performance.

### TruthfulQA

- **Tasks:** 817 questions designed to trigger common misconceptions
- **Format:** Multiple choice (mc2 variant)
- **Standard:** 0-shot
- **Metric:** mc2 (weighted accuracy across true/false answers)
- **Baselines:** GPT-4 ≈ 60%, humans ≈ 94%
- **Gotchas:** Models trained on internet data perform worse (learned misconceptions).

## Reasoning Benchmarks

### HellaSwag

- **Tasks:** Commonsense sentence completion
- **Format:** 4-choice
- **Standard:** 10-shot
- **Metric:** acc_norm (length-normalized)
- **Baselines:** GPT-4 ≈ 95%, Llama 3.1 8B ≈ 82%
- **Gotchas:** Near-saturated for large models. Use acc_norm (not acc).

### ARC-Challenge

- **Tasks:** Grade-school science (challenge subset)
- **Format:** 4-5 choice
- **Standard:** 25-shot
- **Metric:** acc_norm
- **Baselines:** GPT-4 ≈ 96%, Llama 3.1 8B ≈ 79%

### Winogrande

- **Tasks:** Coreference resolution
- **Format:** 2-choice
- **Standard:** 5-shot
- **Metric:** accuracy
- **Baselines:** GPT-4 ≈ 87%, humans ≈ 94%

### BBH (Big Bench Hard)

- **Tasks:** 23 challenging BIG-Bench tasks
- **Standard:** 3-shot CoT
- **Metric:** accuracy
- **Gotchas:** Requires CoT prompting to show benefit.

## Math Benchmarks

### GSM8K

- **Tasks:** 1,319 grade-school math word problems (multi-step)
- **Standard:** 5-shot CoT (or 8-shot)
- **Metric:** exact_match (final numeric answer)
- **Baselines:** GPT-4o ≈ 95%, Llama 3.1 8B ≈ 56%
- **Gotchas:** Answer extraction is critical — must parse final number. Use `exact_match,strict-match` or `exact_match,flexible-extract`.

### MATH (Minerva)

- **Tasks:** Competition math (AMC, AIME level)
- **Standard:** 4-shot
- **Metric:** exact_match
- **Baselines:** GPT-4 ≈ 52%, Llama 3.1 70B ≈ 42%
- **Gotchas:** Much harder than GSM8K. Requires LaTeX parsing for equivalence.

## Code Benchmarks

### HumanEval

- **Tasks:** 164 Python coding problems
- **Standard:** 0-shot (generate function body)
- **Metric:** pass@1 (functional correctness via test execution)
- **Baselines:** GPT-4 ≈ 87%, Code Llama 34B ≈ 48%
- **Gotchas:** Requires code execution sandbox. Temperature=0.0 for pass@1, temperature=0.8 for pass@10/100.

### MBPP (Mostly Basic Python Problems)

- **Tasks:** 974 short Python problems
- **Standard:** 3-shot
- **Metric:** pass@1
- **Baselines:** GPT-4 ≈ 83%

## Instruction Following

### IFEval

- **Tasks:** Verifiable instruction-following (e.g., "write exactly 3 paragraphs")
- **Standard:** 0-shot
- **Metric:** strict accuracy + loose accuracy
- **Baselines:** GPT-4 ≈ 87%, Llama 3.1 70B ≈ 83%
- **Gotchas:** Requires programmatic verification (built into lm-eval).

## Open LLM Leaderboard v2 Tasks

The HuggingFace Open LLM Leaderboard v2 uses:
1. **MMLU-Pro** (knowledge)
2. **GPQA** (graduate-level QA)
3. **MuSR** (multi-step reasoning)
4. **MATH-Hard** (competition math)
5. **IFEval** (instruction following)
6. **BBH** (hard reasoning)

```bash
lm_eval --tasks mmlu_pro,gpqa_main_zeroshot,musr,minerva_math_hard,ifeval,bbh \
  --model vllm --model_args pretrained=my-model \
  --batch_size auto --output_path ./leaderboard_results
```

## Scoring Pitfalls

1. **Few-shot count matters** — MMLU 0-shot vs 5-shot can differ by 10+ points
2. **Normalization** — `acc_norm` vs `acc` can differ significantly on HellaSwag
3. **Answer parsing** — GSM8K scores vary widely with different extraction regex
4. **Chat template** — instruction-tuned models need `--apply_chat_template`
5. **Contamination** — models may have seen benchmark data in training
6. **Sample limit** — using `--limit` gives noisy estimates; report full evals for papers
