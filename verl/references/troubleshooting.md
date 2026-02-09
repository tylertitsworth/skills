# verl Troubleshooting

## Reward Hacking and Training Instability

### Reward increases but eval quality drops

The model is exploiting the reward function rather than genuinely improving.

**Fixes:**
1. **Increase KL penalty** to keep policy close to the reference:
   ```bash
   algorithm.kl_ctrl.kl_coef=0.01  # increase from default 0.001
   ```

2. **Use a better reward function**: Rule-based rewards are exploitable. Consider combining multiple signals:
   ```python
   def compute_reward(data_source, solution_str, ground_truth, extra_info=None):
       correctness = check_answer(solution_str, ground_truth)
       # Add format/quality checks to make reward harder to game
       if len(solution_str) < 10:
           return 0.0  # penalize degenerate short responses
       return correctness
   ```

3. **Clip rewards**: Prevent extreme reward values:
   ```python
   reward = max(-1.0, min(1.0, raw_reward))
   ```

### Loss oscillating or not decreasing

1. **Lower actor learning rate**: Start with `1e-6`, not `1e-5`
2. **Reduce clip ratio**: `actor_rollout_ref.actor.clip_ratio=0.1`
3. **Increase mini-batch size** for more stable gradients
4. **Check reward distribution**: If all rewards are 0, the model has no signal to learn from

### KL divergence exploding

The policy is diverging too far from the reference.

```bash
# Increase KL penalty
algorithm.kl_ctrl.kl_coef=0.05

# Or use KL in loss (GRPO-style) instead of in reward
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.01
algorithm.use_kl_in_reward=False
```

## OOM Issues

### OOM during rollout (vLLM)

vLLM and the training model share GPU memory. Reduce vLLM's allocation:

```bash
actor_rollout_ref.rollout.gpu_memory_utilization=0.3  # reduce from 0.5
actor_rollout_ref.rollout.max_num_seqs=64             # fewer concurrent sequences
```

### OOM during training

```bash
# Reduce micro-batch size
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
critic.ppo_micro_batch_size_per_gpu=1

# Enable gradient checkpointing
actor_rollout_ref.model.enable_gradient_checkpointing=True

# Reduce max sequence length
data.max_response_length=256
```

### OOM with large models

```bash
# Use tensor parallelism for rollout
actor_rollout_ref.rollout.tensor_model_parallel_size=4

# Use more GPUs
trainer.n_gpus_per_node=8
```

## vLLM Rollout Issues

### "CUDA error" or communication error during rollout

The hybrid engine switches between training (FSDP) and rollout (vLLM) modes. Issues often arise during the transition.

```bash
# Disable CUDA graphs for debugging
actor_rollout_ref.rollout.enforce_eager=True

# Reduce GPU memory utilization
actor_rollout_ref.rollout.gpu_memory_utilization=0.3
```

### Rollout is very slow

1. **Increase tensor parallelism** for rollout:
   ```bash
   actor_rollout_ref.rollout.tensor_model_parallel_size=2
   ```

2. **Increase GPU memory for vLLM** (allows larger batch):
   ```bash
   actor_rollout_ref.rollout.gpu_memory_utilization=0.6
   ```

3. **Reduce response length**:
   ```bash
   data.max_response_length=512  # shorter generations are faster
   ```

### Model generating empty or degenerate responses

- Check `rollout.temperature` isn't 0 (no sampling) or too high (>1.5)
- Verify the chat template is correct for your model
- Check that `data.max_response_length` is long enough

## Checkpoint and Merging Issues

### "KeyError" when loading checkpoint

Checkpoint was saved with different config (e.g., different number of GPUs).

**Fix**: Use the model merger to convert to HuggingFace format:
```bash
python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir checkpoints/project/run/global_step_N/actor \
  --target_dir ./merged_hf_model
```

### Merged model produces bad output

Merging may fail silently. Verify:
1. Compare merged model size to original (should be similar)
2. Test with a simple prompt before deploying
3. Ensure `--backend` matches training backend (fsdp or megatron)

## Data Issues

### "KeyError: 'prompt'" 

Data doesn't have the expected column:
```bash
data.prompt_key=question  # use the actual column name
```

### Reward function not being called

Ensure the reward function path and name are correct:
```bash
reward_model.reward_fn.path=/path/to/my_reward.py
reward_model.reward_fn.name=compute_reward  # function name
```

### Validation score always 0

Check that:
1. Validation data has the same format as training data
2. Validation uses greedy decoding (default): `rollout.val_kwargs.temperature=0`
3. Reward function handles the validation data correctly

## Multi-Node Issues

### Nodes can't communicate

```bash
# Check Ray cluster connectivity (verl uses Ray for orchestration)
ray status

# Verify nodes can reach each other
ping $HEAD_NODE
```

### Inconsistent behavior across nodes

All nodes must have:
- Same verl version
- Same model weights (or shared filesystem)
- Same Python/CUDA versions
- Access to the data files
