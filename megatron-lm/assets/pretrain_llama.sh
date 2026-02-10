#!/usr/bin/env bash
# Megatron-LM pretraining launch script for a Llama-style model.
# Designed for 8 GPUs (single node). Adjust parallelism for multi-node.
#
# Prerequisites:
#   - Preprocessed data at DATA_PATH (.bin + .idx files)
#   - Tokenizer from HuggingFace
#   - NVIDIA container with Megatron-LM installed
#
# Usage: bash pretrain_llama.sh
set -euo pipefail

# --- Model Architecture (Llama 3.2 1B-like) ---
NUM_LAYERS=16
HIDDEN_SIZE=2048
NUM_ATTENTION_HEADS=16
NUM_QUERY_GROUPS=4          # GQA
FFN_HIDDEN_SIZE=5632
SEQ_LENGTH=4096

# --- Parallelism (8 GPUs, single node) ---
TP=2                        # Tensor parallel (within NVLink domain)
PP=1                        # Pipeline parallel
# DP = total_gpus / (TP * PP) = 4

# --- Training ---
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=64        # Must be divisible by DP * micro_batch * gradient_accum
TRAIN_ITERS=50000
LR=3e-4
MIN_LR=3e-5
WARMUP_ITERS=2000

# --- Paths ---
DATA_PATH="/data/preprocessed/my_dataset_text_document"
TOKENIZER="meta-llama/Llama-3.2-1B"
CHECKPOINT_DIR="/checkpoints/llama-1b"
LOG_DIR="/logs/llama-1b"

torchrun \
  --nproc_per_node=8 \
  --nnodes=1 \
  pretrain_gpt.py \
  --num-layers ${NUM_LAYERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --num-attention-heads ${NUM_ATTENTION_HEADS} \
  --num-query-groups ${NUM_QUERY_GROUPS} \
  --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
  --seq-length ${SEQ_LENGTH} \
  --max-position-embeddings ${SEQ_LENGTH} \
  --swiglu \
  --normalization RMSNorm \
  --use-rotary-position-embeddings \
  --untie-embeddings-and-output-weights \
  --disable-bias-linear \
  --tensor-model-parallel-size ${TP} \
  --pipeline-model-parallel-size ${PP} \
  --sequence-parallel \
  --use-distributed-optimizer \
  --overlap-grad-reduce \
  --overlap-param-gather \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_ITERS} \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style cosine \
  --lr-warmup-iters ${WARMUP_ITERS} \
  --weight-decay 0.1 \
  --clip-grad 1.0 \
  --bf16 \
  --attention-softmax-in-fp32 \
  --accumulate-allreduce-grads-in-fp32 \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model ${TOKENIZER} \
  --data-path ${DATA_PATH} \
  --split 99,1,0 \
  --save ${CHECKPOINT_DIR} \
  --load ${CHECKPOINT_DIR} \
  --save-interval 1000 \
  --ckpt-format torch_dist \
  --auto-detect-ckpt-format \
  --log-interval 10 \
  --eval-interval 500 \
  --eval-iters 20 \
  --recompute-granularity selective \
  --tensorboard-dir ${LOG_DIR}
