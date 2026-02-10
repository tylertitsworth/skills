---
name: verl
description: >
  Train LLMs with reinforcement learning using verl (Volcano Engine RL) — supporting PPO,
  GRPO, DAPO, RLOO, and more. Use when: (1) Running RLHF or GRPO post-training on language
  models, (2) Configuring reward models or rule-based reward functions, (3) Setting up PPO
  training with actor/critic/reference models, (4) Using GRPO for reasoning model training,
  (5) Configuring rollout generation with vLLM or SGLang, (6) Scaling RL training across
  multiple GPUs with FSDP or Megatron-LM, (7) Writing custom reward functions, (8) Debugging
  RL training issues (reward hacking, KL divergence, OOM).
---

# verl (Volcano Engine Reinforcement Learning)

verl is a production-ready RL training framework for LLMs. Supports PPO, GRPO, DAPO, RLOO, REINFORCE++, ReMax, and more. Uses FSDP/Megatron-LM for training and vLLM/SGLang for rollout generation.

**GitHub**: [verl-project/verl](https://github.com/verl-project/verl) | **Docs**: [verl.readthedocs.io](https://verl.readthedocs.io)

**Requirements**: PyTorch 2.4+, FSDP or Megatron-LM, vLLM or SGLang for rollout generation. Container image: `verl`.

## Core Architecture

verl uses a **3D-HybridEngine** where the same GPU pool switches between:
1. **Rollout** (generation) — uses vLLM/SGLang for fast batched inference
2. **Training** (policy update) — uses FSDP/Megatron-LM for gradient computation
3. **Reference model** — computes reference log-probs for KL penalty

The hybrid engine eliminates memory redundancy by resharding model weights between training (FSDP sharded) and inference (vLLM tensor-parallel) phases on the same GPUs.

**Resource allocation lifecycle per training step:**
1. Load model weights into vLLM, generate rollouts for the batch
2. Offload vLLM, load FSDP-sharded weights for training
3. Compute advantages, run PPO/GRPO updates
4. Save checkpoint, repeat

## SFT (Supervised Fine-Tuning)

verl includes an SFT trainer as a pre-RL step. Run SFT before RL training:

```yaml
# sft_config.yaml
data:
  train_files: /data/sft_train.parquet
  val_files: /data/sft_val.parquet
  prompt_key: prompt
  response_key: response
  micro_batch_size_per_gpu: 4

model:
  path: Qwen/Qwen2.5-7B-Instruct

trainer:
  n_gpus_per_node: 4
  nnodes: 1
  total_epochs: 3
  save_freq: 500
  project_name: my-sft
  logger: ["console", "wandb"]

optim:
  lr: 2e-5
  weight_decay: 0.01
  warmup_steps_proportion: 0.1
  lr_scheduler: cosine
```

SFT data format — Parquet with `prompt` and `response` columns:

```python
# Data prep
data = [{"prompt": tokenizer.apply_chat_template(msgs, tokenize=False),
         "response": answer} for msgs, answer in dataset]
pd.DataFrame(data).to_parquet("sft_train.parquet")
```

## PPO Configuration

PPO is the full RLHF algorithm with actor, critic, reference model, and reward model:

```yaml
# ppo_config.yaml
data:
  train_files: /data/gsm8k/train.parquet
  val_files: /data/gsm8k/test.parquet
  train_batch_size: 256
  max_prompt_length: 512
  max_response_length: 512

actor_rollout_ref:
  model:
    path: Qwen/Qwen2.5-7B-Instruct
  actor:
    optim:
      lr: 1e-6
    ppo_mini_batch_size: 64
    ppo_micro_batch_size_per_gpu: 4
    ppo_epochs: 1
    clip_ratio: 0.2
    grad_clip: 1.0
    entropy_coeff: 0.0
    use_torch_compile: true
  rollout:
    name: vllm
    tensor_model_parallel_size: 1
    gpu_memory_utilization: 0.4
    temperature: 1.0
    top_p: 1.0
    n: 1                              # 1 response per prompt for PPO
  ref:
    log_prob_micro_batch_size_per_gpu: 4

critic:
  model:
    path: Qwen/Qwen2.5-7B-Instruct    # same or separate critic model
  optim:
    lr: 1e-5
  ppo_micro_batch_size_per_gpu: 4

algorithm:
  adv_estimator: gae                   # Generalized Advantage Estimation
  kl_ctrl:
    kl_coef: 0.001

trainer:
  n_gpus_per_node: 4
  nnodes: 1
  total_epochs: 15
  save_freq: 10
  logger: ["console", "wandb"]
  project_name: ppo-gsm8k
```

## GRPO Configuration

GRPO is simpler than PPO — no critic model. Samples multiple responses per prompt and uses group-relative rewards:

```yaml
# grpo_config.yaml
data:
  train_files: /data/gsm8k/train.parquet
  val_files: /data/gsm8k/test.parquet
  train_batch_size: 128
  max_prompt_length: 512
  max_response_length: 1024

actor_rollout_ref:
  model:
    path: Qwen/Qwen2.5-7B-Instruct
  actor:
    optim:
      lr: 1e-6
    use_kl_loss: true                  # KL penalty in loss function
    kl_loss_coef: 0.001
    ppo_mini_batch_size: 64
    ppo_micro_batch_size_per_gpu: 2
  rollout:
    name: vllm
    n: 8                               # 8 responses per prompt (key for GRPO)
    temperature: 1.0
    tensor_model_parallel_size: 2
    gpu_memory_utilization: 0.5

algorithm:
  adv_estimator: grpo                  # group-relative advantage

# No critic section needed — GRPO is critic-free

trainer:
  n_gpus_per_node: 4
  total_epochs: 20
```

## Reward Functions

### Rule-Based Rewards

```python
# verl/utils/reward_score/my_reward.py
import re

def compute_reward(data_source, solution_str, ground_truth, extra_info=None):
    """
    Args:
        data_source: dataset name
        solution_str: model's generated response
        ground_truth: expected answer from dataset
        extra_info: additional metadata
    Returns:
        float: reward score
    """
    # Extract answer from model output (e.g., after "####")
    match = re.search(r"####\s*(-?\d+)", solution_str)
    if match is None:
        return 0.0  # no answer found

    predicted = match.group(1).strip()
    expected = str(ground_truth).strip()

    if predicted == expected:
        return 1.0
    return 0.0
```

Register in config:
```bash
reward_model.reward_fn.path=verl/utils/reward_score/my_reward.py \
reward_model.reward_fn.name=compute_reward
```

### Reward Model (Learned)

Use a trained reward model instead of rules:

```bash
reward_model.enable=True \
reward_model.model.path=my-org/reward-model-7b \
reward_model.micro_batch_size_per_gpu=4
```

### Multi-Reward Composition

```python
def compute_reward(data_source, solution_str, ground_truth, extra_info=None):
    correctness = check_answer(solution_str, ground_truth)  # 0 or 1
    format_score = check_format(solution_str)               # 0 to 0.5
    length_penalty = -max(0, len(solution_str) - 2000) / 10000  # penalize verbose
    return correctness + format_score + length_penalty
```

## Configuration Reference

### Data Config

| Parameter | Purpose | Default |
|-----------|---------|---------|
| `data.train_batch_size` | Global batch size per step | 1024 |
| `data.max_prompt_length` | Max input tokens | 512 |
| `data.max_response_length` | Max generated tokens | 512 |
| `data.prompt_key` | Column name for prompts | `prompt` |

### Actor Config

| Parameter | Purpose | Default |
|-----------|---------|---------|
| `actor_rollout_ref.model.path` | HuggingFace model | required |
| `actor_rollout_ref.actor.optim.lr` | Actor learning rate | 1e-6 |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | PPO mini-batch (global) | 256 |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | Micro-batch per GPU | 8 |
| `actor_rollout_ref.actor.grad_clip` | Gradient clipping | 1.0 |
| `actor_rollout_ref.actor.clip_ratio` | PPO clip ratio | 0.2 |
| `actor_rollout_ref.actor.ppo_epochs` | PPO epochs per step | 1 |
| `actor_rollout_ref.actor.entropy_coeff` | Entropy bonus | 0.0 |
| `actor_rollout_ref.actor.use_kl_loss` | KL loss (for GRPO) | False |
| `actor_rollout_ref.actor.kl_loss_coef` | KL loss coefficient | 0.001 |
| `actor_rollout_ref.actor.use_torch_compile` | torch.compile | True |

### Rollout Config (vLLM)

| Parameter | Purpose | Default |
|-----------|---------|---------|
| `rollout.name` | Engine (vllm, sglang, hf) | vllm |
| `rollout.temperature` | Sampling temperature | 1.0 |
| `rollout.top_p` | Nucleus sampling | 1.0 |
| `rollout.n` | Responses per prompt | 1 (8+ for GRPO) |
| `rollout.tensor_model_parallel_size` | TP for rollout | 1 |
| `rollout.gpu_memory_utilization` | vLLM GPU memory fraction | 0.5 |
| `rollout.enforce_eager` | Disable CUDA graphs | True |

### Algorithm Config

| Parameter | Purpose | Options |
|-----------|---------|---------|
| `algorithm.adv_estimator` | Advantage estimation | gae, grpo, rloo, reinforce_plus_plus |
| `algorithm.kl_ctrl.kl_coef` | KL penalty coefficient | 0.001 |
| `algorithm.use_kl_in_reward` | KL in reward vs loss | True/False |

### Trainer Config

| Parameter | Purpose | Default |
|-----------|---------|---------|
| `trainer.n_gpus_per_node` | GPUs per node | 8 |
| `trainer.nnodes` | Number of nodes | 1 |
| `trainer.total_epochs` | Training epochs | 1 |
| `trainer.save_freq` | Checkpoint frequency (steps) | -1 |
| `trainer.test_freq` | Validation frequency | -1 |
| `trainer.logger` | Logging backends | console |
| `trainer.project_name` | wandb project | verl |
| `trainer.experiment_name` | wandb run name | default |

## Supported Algorithms

| Algorithm | Advantage Estimator | Critic? | Key Feature |
|-----------|-------------------|---------|-------------|
| **PPO** | `gae` | Yes | Full RLHF with value function |
| **GRPO** | `grpo` | No | Group-relative rewards, simpler |
| **DAPO** | `grpo` + DAPO recipe | No | SOTA reasoning (AIME 50pts) |
| **RLOO** | `rloo` | No | Leave-one-out baseline |
| **REINFORCE++** | `reinforce_plus_plus` | No | Improved REINFORCE |
| **ReMax** | `remax` | No | Max-reward baseline |

## Multi-GPU Scaling

### FSDP Backend (Default)

```bash
# 4 GPUs on 1 node
trainer.n_gpus_per_node=4 trainer.nnodes=1

# 8 GPUs on 2 nodes
trainer.n_gpus_per_node=4 trainer.nnodes=2

# With tensor parallelism for large models (rollout)
actor_rollout_ref.rollout.tensor_model_parallel_size=4
```

### Megatron-LM Backend

For very large models (70B+):

```bash
actor_rollout_ref.actor.strategy=megatron \
actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2
```

## Checkpointing

```bash
# Save checkpoints every 10 steps
trainer.save_freq=10

# Checkpoints saved to:
# checkpoints/{project_name}/{experiment_name}/global_step_{N}/actor/

# Merge to HuggingFace format
python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir checkpoints/my_project/my_run/global_step_100/actor \
  --target_dir ./merged_model/huggingface
```

## Data Preparation

verl expects Parquet files with at minimum a `prompt` column:

```python
import pandas as pd

data = []
for item in raw_dataset:
    # Format prompt with chat template
    messages = [
        {"role": "system", "content": "Solve the math problem step by step."},
        {"role": "user", "content": item["question"]},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    data.append({
        "prompt": prompt,
        "answer": item["answer"],          # for reward computation
        "data_source": "gsm8k",            # passed to reward function
    })

df = pd.DataFrame(data)
df.to_parquet("train.parquet")
```

## Advanced Algorithms

Beyond PPO and GRPO, verl supports:

| Algorithm | Config Key | Description |
|---|---|---|
| DAPO | `algorithm.adv_estimator=grpo` + DAPO config | Decoupled clip + dynamic sampling |
| SPIN | Recipe | Self-Play Fine-Tuning |
| SPPO | Recipe | Self-Play Preference Optimization |
| OPO | Recipe | On-Policy RL with Optimal Reward Baseline |
| GPG | Recipe | Group Policy Gradient |

See [verl recipes](https://github.com/verl-project/verl-recipe) for full implementations.

### DAPO Configuration

DAPO uses separated clip epsilons (asymmetric clipping) and dynamic sampling for SOTA reasoning:

```yaml
actor_rollout_ref:
  actor:
    clip_ratio_low: 0.2     # ε_low for lower bound clipping
    clip_ratio_high: 0.28   # ε_high for upper bound clipping (higher = more exploration)
```

The asymmetric clip prevents overly conservative updates while maintaining stability. DAPO on Qwen2.5-32B achieves 50% on AIME 2024.

### Online DPO

Extend verl's PPO infrastructure for online DPO — generate N responses per prompt, score them, form pairwise preferences, then train with DPO loss using a reference policy. See `docs/advance/dpo_extension.md` in the verl repo.

## LoRA Training

```yaml
actor_rollout_ref:
  actor:
    peft:
      peft_type: lora
      lora_rank: 16
      lora_alpha: 32
      target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

## Multi-Turn Rollout

Enable multi-turn conversation training with tool use:

```yaml
rollout:
  multi_turn:
    enable: true
    max_turns: 5
    tool_server: "http://tool-server:8080"
```

## FP8 Rollout

Enable FP8 for faster generation:

```yaml
rollout:
  dtype: fp8
```

## Async Training

verl supports async variants for higher throughput:
- **One-step off-policy**: Generate with previous policy, train with current
- **Fully async**: Overlap generation and training completely

## Monitoring

verl logs to WandB by default. Enable Prometheus + Grafana for rollout monitoring:

```yaml
trainer:
  prometheus:
    enable: true
    port: 9090
```

## Debugging

See `references/troubleshooting.md` for:
- Reward hacking and training instability
- KL divergence explosion
- OOM during rollout or training
- vLLM rollout failures
- Checkpoint and model merging issues

## Cross-References

- [hydra](../hydra/) — Hydra configuration framework (verl uses OmegaConf/Hydra for config composition)
- [wandb](../wandb/) — Experiment tracking and monitoring
- [vllm](../vllm/) — vLLM rollout engine for inference during RL training
- [fsdp](../fsdp/) — FSDP training backend
- [megatron-lm](../megatron-lm/) — Megatron-LM training backend for large models
- [pytorch](../pytorch/) — PyTorch distributed training fundamentals

## Reference

- [verl docs](https://verl.readthedocs.io/)
- [verl GitHub](https://github.com/verl-project/verl)
- [verl recipes](https://github.com/verl-project/verl-recipe)
- [HybridFlow paper (EuroSys 2025)](https://arxiv.org/abs/2409.19256)
- `references/troubleshooting.md` — common errors and fixes
- `assets/architecture.md` — Mermaid architecture diagrams
