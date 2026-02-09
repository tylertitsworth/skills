# Megatron-LM Troubleshooting

## Communication Errors

### Rank crashed ("remote process exited")

One rank crashed — check pod logs across all replicas. Common causes:
1. OOM on one rank — check container exit codes and resource limits
2. GPU hardware error — check node-level DCGM exporter metrics
3. Network partition between nodes — verify pod-to-pod connectivity and Service endpoints

### Hang during initialization

All ranks must reach `torch.distributed.init_process_group` simultaneously. Verify:
- `MASTER_ADDR` and `MASTER_PORT` env vars are correctly set in the pod spec
- Network policies allow inter-pod communication on the master port (default 29500)
- All pods are scheduled and running before the timeout expires

## OOM Issues

### OOM during forward pass

1. **Reduce micro-batch size**: `--micro-batch-size 1`
2. **Enable activation checkpointing**: `--recompute-activations`
3. **Increase TP** to shard layers more: `--tensor-model-parallel-size 8`
4. **Enable sequence parallelism**: `--sequence-parallel`

### OOM during backward pass

Gradients and optimizer states dominate memory at scale:
1. **Use distributed optimizer**: `--use-distributed-optimizer`
2. **Overlap communication**: `--overlap-grad-reduce` (reduces peak memory)
3. **Increase PP** to reduce per-stage layer count

### OOM with long sequences

```bash
# Use context parallelism for long sequences
--context-parallel-size 2

# Or reduce sequence length during initial training
--seq-length 2048
```

### Estimating memory requirements

```
Per-GPU memory ≈ 
  model_params_per_gpu × bytes_per_param (bf16 = 2)
  + optimizer_states_per_gpu (AdamW = 12 bytes/param without ZeRO)
  + activations (depends on batch size, seq length, recompute)
  + communication buffers

With distributed optimizer: optimizer_states / DP_size
With TP=N: model_params / N (approx)
```

## Pipeline Bubble Optimization

Pipeline parallelism introduces "bubbles" (idle time) at pipeline boundaries.

### Reduce bubble fraction

```bash
# Interleaved pipeline schedule (more stages, smaller bubbles)
--num-layers-per-virtual-pipeline-stage 2

# Bubble fraction ≈ (PP - 1) / (micro_batches × PP)
# More micro-batches = smaller bubbles
--global-batch-size 2048  # more micro-batches per step
```

### Check pipeline utilization

Monitor `iteration time` — if it's much higher than expected:
- Increase global batch size for more micro-batches
- Use interleaved schedule
- Reduce PP if possible (increase TP instead)

## Checkpoint Issues

### "ValueError: Could not find checkpoint"

Checkpoint structure must match the parallelism config used during saving:
```
checkpoints/
├── iter_001000/
│   ├── mp_rank_00_000/  # TP rank 0, PP rank 0
│   ├── mp_rank_01_000/  # TP rank 1, PP rank 0
│   ├── mp_rank_00_001/  # TP rank 0, PP rank 1
│   └── ...
```

If changing parallelism config, use Megatron Bridge to convert first.

### Checkpoint too large / slow to save

```bash
# Async saving (doesn't block training)
--async-save

# Save less frequently
--save-interval 5000

# Use distributed checkpointing (torch.distributed.checkpoint)
--use-dist-ckpt
```

### Conversion errors (HuggingFace ↔ Megatron)

- Ensure `--model-type` matches your architecture (llama, gpt, etc.)
- Verify TP/PP sizes match the checkpoint being converted
- Check Megatron Bridge version compatibility with your Megatron Core version

## Data Loading Issues

### "IndexError" during data loading

Binary data files may be corrupted or mismatched:
```bash
# Regenerate preprocessed data
python tools/preprocess_data.py \
    --input data.jsonl \
    --output-prefix my_data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model $MODEL_NAME \
    --workers 32
```

### Data loading is slow

```bash
# Increase data loader workers
--num-workers 8

# Use mmap (default) for memory-mapped access
# Ensure data is on fast storage (NVMe SSD, not NFS)
```

### Tokenizer mismatch

Using a different tokenizer than what the data was preprocessed with produces garbage:
```bash
# Tokenizer must match between preprocessing and training
# Preprocessing:
--tokenizer-type HuggingFaceTokenizer --tokenizer-model meta-llama/Llama-3.1-8B
# Training (must be identical):
--tokenizer-type HuggingFaceTokenizer --tokenizer-model meta-llama/Llama-3.1-8B
```

## Performance Issues

### Low MFU (Model FLOP Utilization)

Expected MFU: 40-50% on H100. If significantly lower:

1. **Enable communication overlap**:
   ```bash
   --overlap-grad-reduce --overlap-param-gather --tp-comm-overlap
   ```

2. **Check parallelism config**: TP should stay within NVLink domain

3. **Increase batch size**: Larger batches = better GPU utilization

4. **Profile**:
   ```bash
   --profile --profile-step-start 10 --profile-step-end 12
   # Creates a Chrome trace in the log directory
   ```

### Flash Attention not being used

```bash
# Ensure flash attention is installed
# Add to container image: flash-attn (build from source)

# Enable in config
--attention-implementation flash_attention_2
```
