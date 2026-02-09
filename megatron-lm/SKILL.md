---
name: megatron-lm
description: >
  Train large language models at scale with NVIDIA Megatron-LM and Megatron Core. Use when:
  (1) Configuring tensor, pipeline, sequence, or context parallelism, (2) Training models
  from billions to hundreds of billions of parameters, (3) Setting up mixed precision (BF16,
  FP8) training, (4) Managing Megatron checkpoints and converting to/from HuggingFace,
  (5) Configuring data loading and tokenization, (6) Training MoE (Mixture of Experts) models,
  (7) Optimizing communication overlap and memory efficiency, (8) Debugging distributed
  training issues at scale.
---

# Megatron-LM

NVIDIA Megatron-LM is the framework for training transformer models at scale. It provides GPU-optimized building blocks with advanced parallelism strategies. Achieves up to 47% MFU on H100 clusters.

**Components:**
- **Megatron Core** — composable library with parallelism, kernels, model building blocks
- **Megatron-LM** — reference training scripts using Megatron Core
- **Megatron Bridge** — HuggingFace ↔ Megatron checkpoint conversion

## Setup

```bash
# Install Megatron Core
pip install --no-build-isolation megatron-core[mlm,dev]

# Clone for training scripts
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install --no-build-isolation .[mlm,dev]
```

**Requirements**: NVIDIA GPUs (A100/H100), CUDA 12+, PyTorch 2.4+, NCCL, Apex (optional for FP8).

## Parallelism Strategies

Megatron combines multiple parallelism dimensions. Total GPUs = TP × PP × DP (× CP × EP for advanced configs).

### Tensor Parallelism (TP)

Splits individual layers (attention heads, MLP columns) across GPUs within a node:

```bash
--tensor-model-parallel-size 4  # shard layers across 4 GPUs
```

Best for: Intra-node (requires fast NVLink). Typical values: 1, 2, 4, 8.

### Pipeline Parallelism (PP)

Splits model layers across groups of GPUs:

```bash
--pipeline-model-parallel-size 4  # 4 pipeline stages
--num-layers-per-virtual-pipeline-stage 2  # interleaved schedule
```

Best for: Inter-node or when TP alone isn't enough. The interleaved schedule reduces pipeline bubbles.

### Data Parallelism (DP)

Replicates the model across GPU groups — standard gradient-synchronized training. DP size is computed automatically:
```
DP = total_GPUs / (TP × PP)
```

### Sequence Parallelism (SP)

Distributes sequence-dimension operations (LayerNorm, Dropout) across TP ranks. Requires TP > 1:

```bash
--tensor-model-parallel-size 4
--sequence-parallel  # enable SP (requires TP > 1)
```

### Context Parallelism (CP)

Splits long sequences across GPUs — enables training with very long contexts:

```bash
--context-parallel-size 2  # split context across 2 GPUs
```

### Expert Parallelism (EP)

For MoE models — distributes experts across GPUs:

```bash
--expert-model-parallel-size 4  # distribute experts across 4 GPUs
--num-experts 64
--moe-router-topk 2
```

## Training Script

### GPT Pretraining Example

```bash
#!/bin/bash

GPUS_PER_NODE=8
NNODES=4
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))

# Parallelism config
TP=4
PP=2
DP=$((WORLD_SIZE / (TP * PP)))

# Model args (7B-like)
MODEL_ARGS=(
    --num-layers 32
    --hidden-size 4096
    --num-attention-heads 32
    --seq-length 4096
    --max-position-embeddings 4096
    --ffn-hidden-size 11008
    --num-query-groups 8          # GQA (grouped query attention)
    --swiglu                      # SwiGLU activation
    --normalization RMSNorm
    --use-rotary-position-embeddings
    --no-position-embedding       # RoPE replaces absolute
    --untie-embeddings-and-output-weights
    --disable-bias-linear
)

# Parallelism args
PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP
    --pipeline-model-parallel-size $PP
    --sequence-parallel
    --use-distributed-optimizer   # ZeRO-1 style optimizer sharding
    --overlap-grad-reduce         # overlap gradient all-reduce with backward
    --overlap-param-gather        # overlap parameter all-gather with forward
)

# Training args
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 1024
    --lr 3e-4
    --min-lr 3e-5
    --lr-decay-style cosine
    --lr-warmup-iters 2000
    --train-iters 100000
    --weight-decay 0.1
    --clip-grad 1.0
    --bf16
    --attention-softmax-in-fp32
    --accumulate-allreduce-grads-in-fp32
)

# Data args
DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model meta-llama/Llama-3.1-8B
    --data-path $DATA_PATH
    --split 99,1,0
    --dataloader-type cyclic
)

# Checkpointing
CHECKPOINT_ARGS=(
    --save $CHECKPOINT_DIR
    --load $CHECKPOINT_DIR
    --save-interval 1000
)

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    pretrain_gpt.py \
    "${MODEL_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}"
```

## Mixed Precision

### BF16 (Standard)

```bash
--bf16
--attention-softmax-in-fp32          # numerical stability
--accumulate-allreduce-grads-in-fp32 # gradient precision
```

### FP8 (H100/Blackwell)

```bash
--fp8-format hybrid        # E4M3 for forward, E5M2 for backward
--fp8-amax-history-len 1024
--fp8-amax-compute-algo max
```

## Communication Optimization

```bash
# Overlap gradient all-reduce with backward pass
--overlap-grad-reduce

# Overlap parameter all-gather with forward pass
--overlap-param-gather

# Overlap TP communication with compute
--tp-comm-overlap

# Bucket size for gradient reduction (tune for your network)
--dp-comm-overlap-bucket-size 100000000
```

## Data Preparation

Megatron uses a custom binary format for efficient data loading:

```bash
# Preprocess data to Megatron format
python tools/preprocess_data.py \
    --input raw_data.jsonl \
    --output-prefix my_dataset \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model meta-llama/Llama-3.1-8B \
    --workers 32 \
    --append-eod

# Creates: my_dataset_text_document.bin, my_dataset_text_document.idx
```

**Blending multiple datasets:**
```bash
--data-path \
    0.7 dataset_a_text_document \
    0.2 dataset_b_text_document \
    0.1 dataset_c_text_document
```

## Checkpointing

### Save and Resume

```bash
--save /checkpoints/my_model
--load /checkpoints/my_model
--save-interval 1000
```

Megatron saves sharded checkpoints — each TP/PP rank saves its shard.

### Convert to HuggingFace

Use [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge):

```bash
pip install megatron-bridge

# Megatron → HuggingFace
python -m megatron.bridge.convert \
    --source megatron \
    --target hf \
    --model-type llama \
    --input-path /checkpoints/my_model/iter_100000 \
    --output-path ./hf_model \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 2

# HuggingFace → Megatron
python -m megatron.bridge.convert \
    --source hf \
    --target megatron \
    --model-type llama \
    --input-path meta-llama/Llama-3.1-8B \
    --output-path ./megatron_checkpoint \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 2
```

## Activation Checkpointing

Trade compute for memory:

```bash
# Recompute all activations (maximum memory savings)
--recompute-activations

# Selective recomputation (better speed/memory tradeoff)
--recompute-granularity selective

# Full recomputation with specific granularity
--recompute-granularity full
--recompute-method uniform
--recompute-num-layers 1
```

## MoE (Mixture of Experts)

```bash
--num-experts 64
--moe-router-topk 2
--moe-aux-loss-coeff 1e-2
--moe-grouped-gemm              # fused expert compute
--expert-model-parallel-size 4  # distribute experts
--moe-permute-fusion            # fused permutation
```

## Distributed Optimizer

ZeRO-1 style — shards optimizer states across DP ranks:

```bash
--use-distributed-optimizer
```

Reduces per-GPU optimizer memory by `1/DP_size`.

## FP8 Training

H100+ GPUs support FP8 for ~30% speedup:

```bash
--fp8-format hybrid           # E4M3 forward, E5M2 backward
--fp8-amax-compute-algo max   # or most_recent
--fp8-amax-history-len 1024
--fp8-margin 0                # scaling margin
```

Requires Transformer Engine (`pip install transformer-engine`).

## Data Preprocessing

Megatron requires pre-tokenized binary datasets:

```bash
# Preprocess with megatron tools
python tools/preprocess_data.py \
  --input raw_data.jsonl \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model meta-llama/Llama-3.1-8B \
  --output-prefix ./data/my_dataset \
  --workers 32 \
  --append-eod

# Produces: my_dataset_text_document.bin + .idx
# Use in training:
--data-path ./data/my_dataset_text_document
```

### Data Blending

```bash
# Weighted blend of multiple datasets
--data-path 0.7 dataset1_text_document 0.3 dataset2_text_document
```

## Megatron Bridge (Checkpoint Conversion)

Convert HuggingFace ↔ Megatron checkpoints:

```bash
pip install megatron-bridge

# HuggingFace → Megatron
python -m megatron.bridge.convert \
  --source-format huggingface \
  --target-format megatron \
  --source-path meta-llama/Llama-3.1-8B \
  --target-path ./megatron_ckpt \
  --target-tp 4 --target-pp 2

# Megatron → HuggingFace
python -m megatron.bridge.convert \
  --source-format megatron \
  --target-format huggingface \
  --source-path ./megatron_ckpt \
  --target-path ./hf_model
```

## Communication Overlap

Enable overlap of computation and communication for higher throughput:

```bash
--overlap-grad-reduce              # overlap gradient all-reduce with backward
--overlap-param-gather             # overlap parameter all-gather with forward
--tp-comm-overlap                  # overlap TP communication with compute
```

## Common Parallelism Configurations

| Model Size | GPUs | TP | PP | DP | Notes |
|-----------|------|----|----|-----|-------|
| 7B | 8 | 1 | 1 | 8 | Single node, DP only |
| 13B | 8 | 2 | 1 | 4 | TP within NVLink domain |
| 70B | 32 | 4 | 2 | 4 | TP + PP across nodes |
| 175B | 128 | 8 | 4 | 4 | Full 3D parallelism |
| 405B | 256 | 8 | 8 | 4 | Large-scale multi-node |

**Rules of thumb:**
- TP ≤ GPUs per node (NVLink bound)
- PP for models exceeding single-node memory
- Increase DP for higher throughput
- SP is free with TP (always enable)

## Debugging

See `references/troubleshooting.md` for:
- Communication errors
- OOM at various scales
- Pipeline bubble optimization
- Checkpoint conversion issues
- Data loading problems

## Reference

- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- [Megatron Core docs](https://docs.nvidia.com/Megatron-Core/)
- [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)
- [Megatron-LM paper](https://arxiv.org/abs/1909.08053)
- `references/troubleshooting.md` — common errors and fixes
