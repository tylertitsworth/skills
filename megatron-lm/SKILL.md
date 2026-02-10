---
name: megatron-lm
description: >
  Train large language models at scale with NVIDIA Megatron-LM and Megatron Core. Use when:
  (1) Configuring tensor, pipeline, sequence, context, or expert parallelism,
  (2) Training models from billions to hundreds of billions of parameters,
  (3) Setting up mixed precision (BF16, FP8) training,
  (4) Managing Megatron checkpoints and converting to/from HuggingFace,
  (5) Configuring data loading and tokenization,
  (6) Training MoE (Mixture of Experts) models,
  (7) Optimizing communication overlap and memory efficiency,
  (8) Configuring resiliency and fault tolerance at scale.
---

# Megatron-LM

NVIDIA Megatron-LM is the framework for training transformer models at scale. Provides GPU-optimized building blocks with advanced parallelism strategies. Achieves up to 47% MFU on H100 clusters.

**Components:**
- **Megatron Core** — composable library with parallelism, kernels, model building blocks
- **Megatron-LM** — reference training scripts using Megatron Core
- **Megatron Bridge** — HuggingFace ↔ Megatron checkpoint conversion

**Requirements**: NVIDIA GPUs (A100/H100), CUDA 12+, PyTorch 2.4+, Apex (optional for FP8).

## Parallelism Configuration

Megatron combines multiple parallelism dimensions. Total GPUs = TP × PP × DP (× CP × EP for advanced configs).

### Parallelism Settings

| Setting | Purpose | Default | Notes |
|---|---|---|---|
| `tensor_model_parallel_size` | Shard layers across GPUs (TP) | `1` | ≤ GPUs per node (NVLink bound) |
| `pipeline_model_parallel_size` | Split layers into pipeline stages (PP) | `1` | For models exceeding single-node memory |
| `sequence_parallel` | Distribute LayerNorm/Dropout across TP ranks (SP) | `False` | Requires TP > 1, always enable with TP |
| `context_parallel_size` | Split long sequences across GPUs (CP) | `1` | For very long context training |
| `expert_model_parallel_size` | Distribute MoE experts across GPUs (EP) | `1` | For MoE models |
| `num_layers_per_virtual_pipeline_stage` | Interleaved PP schedule | None | Reduces pipeline bubbles |
| `use_distributed_optimizer` | ZeRO-1 style optimizer sharding | `False` | Reduces per-GPU optimizer memory by 1/DP |

### Recommended Configurations

| Model Size | GPUs | TP | PP | DP | Notes |
|-----------|------|----|----|-----|-------|
| 7B | 8 | 1 | 1 | 8 | Single node, DP only |
| 13B | 8 | 2 | 1 | 4 | TP within NVLink domain |
| 70B | 32 | 4 | 2 | 4 | TP + PP across nodes |
| 175B | 128 | 8 | 4 | 4 | Full 3D parallelism |
| 405B | 256 | 8 | 8 | 4 | Large-scale multi-node |

## Model Architecture Settings

| Setting | Purpose | Example |
|---|---|---|
| `num_layers` | Transformer layers | `32` |
| `hidden_size` | Hidden dimension | `4096` |
| `num_attention_heads` | Attention heads | `32` |
| `num_query_groups` | GQA groups (0 = MHA) | `8` |
| `ffn_hidden_size` | FFN hidden size | `11008` |
| `seq_length` | Training sequence length | `4096` |
| `max_position_embeddings` | Max position embeddings | `4096` |
| `swiglu` | SwiGLU activation | `True` |
| `normalization` | Norm type (`LayerNorm`, `RMSNorm`) | `RMSNorm` |
| `use_rotary_position_embeddings` | RoPE | `True` |
| `untie_embeddings_and_output_weights` | Separate embed/output weights | `True` |
| `disable_bias_linear` | Remove bias from linear layers | `True` |

## Training Settings

| Setting | Purpose | Default |
|---|---|---|
| `micro_batch_size` | Per-GPU batch size | `1` |
| `global_batch_size` | Total batch across all GPUs | required |
| `lr` | Learning rate | required |
| `min_lr` | Minimum LR (for decay) | `0.0` |
| `lr_decay_style` | Decay schedule (`cosine`, `linear`, `constant`) | `cosine` |
| `lr_warmup_iters` | Warmup iterations | `0` |
| `train_iters` | Total training iterations | required |
| `weight_decay` | Weight decay | `0.01` |
| `clip_grad` | Gradient clipping | `1.0` |
| `dataloader_type` | Dataloader (`cyclic`, `single`) | `cyclic` |
| `split` | Train/val/test split | `99,1,0` |

## Mixed Precision

| Setting | Purpose |
|---|---|
| `bf16` | BF16 training (standard for A100/H100) |
| `attention_softmax_in_fp32` | FP32 softmax for numerical stability |
| `accumulate_allreduce_grads_in_fp32` | FP32 gradient accumulation |
| `fp8_format` | FP8 format: `hybrid` (E4M3 forward, E5M2 backward) |
| `fp8_amax_history_len` | FP8 amax history length | 
| `fp8_amax_compute_algo` | FP8 amax algorithm (`max`, `most_recent`) |

FP8 requires H100+ GPUs and Transformer Engine in the container image.

## Communication Optimization

| Setting | Purpose |
|---|---|
| `overlap_grad_reduce` | Overlap gradient all-reduce with backward pass |
| `overlap_param_gather` | Overlap parameter all-gather with forward pass |
| `tp_comm_overlap` | Overlap TP communication with compute |
| `dp_comm_overlap_bucket_size` | Bucket size for gradient reduction (tune for network) |

## Memory Optimization

| Setting | Purpose |
|---|---|
| `recompute_activations` | Recompute all activations (max memory savings) |
| `recompute_granularity` | `selective` (recommended) or `full` |
| `recompute_method` | `uniform` or `block` (with `full` granularity) |
| `recompute_num_layers` | Layers to recompute per stage |
| `use_distributed_optimizer` | Shard optimizer state across DP ranks |
| `cpu_offload_optimizer` | Offload optimizer to CPU (slower, saves GPU) |

## MoE (Mixture of Experts)

| Setting | Purpose |
|---|---|
| `num_experts` | Total number of experts |
| `moe_router_topk` | Experts per token |
| `moe_aux_loss_coeff` | Auxiliary load balancing loss coefficient |
| `moe_grouped_gemm` | Fused expert computation |
| `expert_model_parallel_size` | Distribute experts across GPUs |
| `moe_permute_fusion` | Fused permutation kernel |

## Resiliency and Fault Tolerance

### Checkpointing

| Setting | Purpose |
|---|---|
| `save` | Checkpoint save directory (PVC path) |
| `load` | Checkpoint load directory |
| `save_interval` | Save every N iterations |
| `ckpt_format` | Format: `torch` or `torch_dist` (distributed, faster) |
| `auto_detect_ckpt_format` | Auto-detect checkpoint format on load |
| `no_save_optim` | Skip optimizer state in checkpoint (smaller files) |
| `no_save_rng` | Skip RNG state |

Megatron saves sharded checkpoints — each TP/PP rank saves its shard to the same directory.

### Elastic Training and Failure Recovery

- **Checkpoint-based recovery**: Set `save_interval` frequently (every 100-500 iterations). On pod failure, the Job restarts and resumes from the last checkpoint via `load`.
- **torch_dist format**: Use `ckpt_format="torch_dist"` for async distributed checkpointing — significantly faster save/load at scale.
- **Straggler detection**: Set `enable_straggler_detection=True` to log slow ranks. Helps identify failing nodes before they crash.
- **Manual elastic resizing**: Megatron doesn't support automatic elastic training. To change GPU count, save a checkpoint, convert parallelism with Megatron Bridge, and restart.

### Checkpoint Conversion (Megatron Bridge)

Convert between HuggingFace and Megatron formats, or change parallelism dimensions:

```python
# Use as a pre/post-training Job
# HuggingFace → Megatron
# python -m megatron.bridge.convert \
#   --source hf --target megatron \
#   --input-path meta-llama/Llama-3.1-8B \
#   --output-path /checkpoints/megatron \
#   --tensor-model-parallel-size 4 --pipeline-model-parallel-size 2

# Change parallelism dimensions (re-shard)
# python -m megatron.bridge.convert \
#   --source megatron --target megatron \
#   --input-path /checkpoints/tp4_pp2 \
#   --output-path /checkpoints/tp8_pp1 \
#   --source-tp 4 --source-pp 2 \
#   --target-tp 8 --target-pp 1
```

## Data Configuration

| Setting | Purpose |
|---|---|
| `tokenizer_type` | `HuggingFaceTokenizer` or `SentencePieceTokenizer` |
| `tokenizer_model` | HF model ID or path to tokenizer |
| `data_path` | Path to preprocessed binary data (`.bin` + `.idx`) |
| `split` | Train/val/test split ratio (e.g., `99,1,0`) |
| `dataloader_type` | `cyclic` (wraps around) or `single` (one pass) |

### Data Preprocessing

Run as a preprocessing Job before training:

```python
# python tools/preprocess_data.py \
#   --input raw_data.jsonl \
#   --output-prefix /data/my_dataset \
#   --tokenizer-type HuggingFaceTokenizer \
#   --tokenizer-model meta-llama/Llama-3.1-8B \
#   --workers 32 --append-eod
#
# Produces: my_dataset_text_document.bin + .idx
```

**Data blending** — weighted mix of datasets:
```
data_path: "0.7 dataset_a_text_document 0.3 dataset_b_text_document"
```

## Kubernetes Deployment

Megatron training runs as a multi-node PyTorchJob or similar CRD. Key pod spec considerations:

- **`MASTER_ADDR` / `MASTER_PORT`**: Set via the training operator (PyTorchJob sets these automatically)
- **Shared storage**: All ranks need access to the same data and checkpoint PVC
- **`/dev/shm`**: Mount as emptyDir with `medium: Memory` for inter-GPU communication
- **GPU topology**: Use `topologySpreadConstraints` or node affinity to keep TP ranks on the same node
- **`nproc_per_node`**: Match to `nvidia.com/gpu` resource limit

## Debugging

See `references/troubleshooting.md` for:
- Communication errors and rank failures
- OOM at various scales
- Pipeline bubble optimization
- Checkpoint conversion issues
- Data loading problems

## Cross-References

- [pytorch](../pytorch/) — PyTorch distributed training fundamentals
- [fsdp](../fsdp/) — Alternative: FSDP for smaller-scale training
- [flash-attention](../flash-attention/) — Ring Attention / context parallelism
- [aws-efa](../aws-efa/) — EFA networking for multi-node Megatron training
- [verl](../verl/) — RL training using Megatron-LM backend

## Reference

- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- [Megatron Core docs](https://docs.nvidia.com/Megatron-Core/)
- [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)
- `references/troubleshooting.md` — common errors and fixes
- `assets/pretrain_llama.sh` — launch script for Llama-style pretraining with TP, distributed optimizer, selective recomputation, and torch_dist checkpoints
- `assets/architecture.md` — Mermaid architecture diagrams
