#!/usr/bin/env bash
# Disaggregated Prefill-Decode Setup Script
# Launches a prefill instance + decode instance + proxy on a single node.
# Customize MODEL, ports, GPU assignments, and TP sizes for your environment.
#
# Usage: bash disagg_setup.sh
# Stop:  Ctrl+C (cleanup trap kills all child processes)

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PREFILL_PORT="${PREFILL_PORT:-8100}"
DECODE_PORT="${DECODE_PORT:-8200}"
PROXY_PORT="${PROXY_PORT:-8000}"

# GPU assignments (comma-separated CUDA device indices)
PREFILL_GPUS="${PREFILL_GPUS:-0,1}"
DECODE_GPUS="${DECODE_GPUS:-2,3}"

# Parallelism
PREFILL_TP="${PREFILL_TP:-2}"
DECODE_TP="${DECODE_TP:-2}"

# NixlConnector settings
NIXL_PREFILL_PORT="${NIXL_PREFILL_PORT:-5600}"
NIXL_DECODE_PORT="${NIXL_DECODE_PORT:-5601}"

# Memory
PREFILL_MEM="${PREFILL_MEM:-0.90}"
DECODE_MEM="${DECODE_MEM:-0.95}"  # Decode gets more KV cache headroom

# ── Cleanup ────────────────────────────────────────────────────────
PIDS=()
cleanup() {
    echo ""
    echo "Shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo "All processes stopped."
}
trap cleanup EXIT INT TERM

# ── Wait for server ────────────────────────────────────────────────
wait_for_server() {
    local port=$1
    local timeout="${2:-120}"
    echo "Waiting for server on port $port (timeout: ${timeout}s)..."
    for i in $(seq 1 "$timeout"); do
        if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
            echo "  ✓ Server on port $port is ready"
            return 0
        fi
        sleep 1
    done
    echo "  ✗ Server on port $port failed to start within ${timeout}s"
    return 1
}

# ── Launch Prefill Instance ────────────────────────────────────────
echo "Starting prefill instance (GPUs: $PREFILL_GPUS, TP: $PREFILL_TP, port: $PREFILL_PORT)..."
CUDA_VISIBLE_DEVICES="$PREFILL_GPUS" \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT="$NIXL_PREFILL_PORT" \
vllm serve "$MODEL" \
    --port "$PREFILL_PORT" \
    --tensor-parallel-size "$PREFILL_TP" \
    --gpu-memory-utilization "$PREFILL_MEM" \
    --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_producer\"}" \
    &
PIDS+=($!)

# ── Launch Decode Instance ─────────────────────────────────────────
echo "Starting decode instance (GPUs: $DECODE_GPUS, TP: $DECODE_TP, port: $DECODE_PORT)..."
CUDA_VISIBLE_DEVICES="$DECODE_GPUS" \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT="$NIXL_DECODE_PORT" \
vllm serve "$MODEL" \
    --port "$DECODE_PORT" \
    --tensor-parallel-size "$DECODE_TP" \
    --gpu-memory-utilization "$DECODE_MEM" \
    --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_consumer\"}" \
    &
PIDS+=($!)

# ── Wait for Both ─────────────────────────────────────────────────
wait_for_server "$PREFILL_PORT" 180
wait_for_server "$DECODE_PORT" 180

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Disaggregated serving is running"
echo "  Prefill:  http://localhost:$PREFILL_PORT  (GPUs: $PREFILL_GPUS)"
echo "  Decode:   http://localhost:$DECODE_PORT  (GPUs: $DECODE_GPUS)"
echo ""
echo "  Route requests through your proxy/router at port $PROXY_PORT"
echo "  or call prefill → decode manually (see references/disaggregated-serving.md)"
echo "════════════════════════════════════════════════════════════════"

# Keep running until interrupted
wait
