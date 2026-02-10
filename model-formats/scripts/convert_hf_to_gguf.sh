#!/usr/bin/env bash
# Convert a HuggingFace model to GGUF format for use with Ollama/llama.cpp.
#
# Usage:
#   ./convert_hf_to_gguf.sh <hf_model_id_or_path> [output_dir] [quantization]
#
# Examples:
#   ./convert_hf_to_gguf.sh meta-llama/Llama-3.2-1B
#   ./convert_hf_to_gguf.sh ./my-finetuned-model /output q4_K_M
#   ./convert_hf_to_gguf.sh meta-llama/Llama-3.2-1B /output f16
#
# Quantization types: f32, f16, bf16, q8_0, q4_K_M, q4_K_S, q5_K_M, q5_K_S, q6_K, q3_K_M
# Requires: pip install llama-cpp-python (or clone llama.cpp)
set -euo pipefail

MODEL="${1:?Usage: $0 <hf_model_id_or_path> [output_dir] [quantization]}"
OUTPUT_DIR="${2:-./gguf-output}"
QUANT="${3:-q4_K_M}"

mkdir -p "${OUTPUT_DIR}"

MODEL_NAME=$(basename "${MODEL}")

echo "==> Step 1: Convert HF model to GGUF (f16)"
python3 -m llama_cpp.convert \
  --outfile "${OUTPUT_DIR}/${MODEL_NAME}-f16.gguf" \
  --outtype f16 \
  "${MODEL}" 2>/dev/null || \
python3 convert_hf_to_gguf.py "${MODEL}" \
  --outfile "${OUTPUT_DIR}/${MODEL_NAME}-f16.gguf" \
  --outtype f16

if [ "${QUANT}" != "f16" ] && [ "${QUANT}" != "f32" ] && [ "${QUANT}" != "bf16" ]; then
  echo "==> Step 2: Quantize to ${QUANT}"
  llama-quantize \
    "${OUTPUT_DIR}/${MODEL_NAME}-f16.gguf" \
    "${OUTPUT_DIR}/${MODEL_NAME}-${QUANT}.gguf" \
    "${QUANT}"
  echo "==> Quantized model: ${OUTPUT_DIR}/${MODEL_NAME}-${QUANT}.gguf"
else
  echo "==> Output: ${OUTPUT_DIR}/${MODEL_NAME}-f16.gguf"
fi

echo ""
echo "To use with Ollama:"
echo "  echo 'FROM ${OUTPUT_DIR}/${MODEL_NAME}-${QUANT}.gguf' > Modelfile"
echo "  ollama create ${MODEL_NAME} -f Modelfile"
echo "  ollama run ${MODEL_NAME}"
