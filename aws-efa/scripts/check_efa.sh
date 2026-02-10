#!/usr/bin/env bash
# Verify EFA device availability and NCCL configuration on a node.
# Run inside a GPU pod on an EFA-enabled instance.
#
# Usage: bash check_efa.sh
set -euo pipefail

echo "=== EFA Device Check ==="
if [ -d /sys/class/infiniband ]; then
  echo "EFA devices found:"
  ls /sys/class/infiniband/
  for dev in /sys/class/infiniband/*/; do
    dev_name=$(basename "$dev")
    echo "  ${dev_name}:"
    echo "    Port state: $(cat ${dev}ports/1/state 2>/dev/null || echo 'N/A')"
    echo "    Link layer: $(cat ${dev}ports/1/link_layer 2>/dev/null || echo 'N/A')"
  done
else
  echo "WARNING: No /sys/class/infiniband — EFA devices not visible"
  echo "  Check: EFA device plugin running? Pod has 'vpc.amazonaws.com/efa' resource?"
fi

echo ""
echo "=== EFA Resource in Pod ==="
if [ -n "${VPC_EFA_DEVICE:-}" ]; then
  echo "VPC_EFA_DEVICE: ${VPC_EFA_DEVICE}"
else
  echo "VPC_EFA_DEVICE not set (expected when EFA resource allocated)"
fi

echo ""
echo "=== NVIDIA GPU Check ==="
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "nvidia-smi not found"
fi

echo ""
echo "=== Libfabric Provider ==="
if command -v fi_info &>/dev/null; then
  echo "Available providers:"
  fi_info -p efa 2>/dev/null | head -20 || echo "  EFA provider not found in fi_info"
  echo ""
  echo "Libfabric version: $(fi_info --version 2>/dev/null | head -1)"
else
  echo "fi_info not found (install libfabric)"
fi

echo ""
echo "=== NCCL Environment ==="
for var in NCCL_DEBUG NCCL_SOCKET_IFNAME NCCL_PROTO NCCL_ALGO FI_PROVIDER FI_EFA_USE_DEVICE_RDMA; do
  echo "  ${var}=${!var:-<not set>}"
done

echo ""
echo "=== Network Interfaces ==="
ip -brief addr show 2>/dev/null | grep -E "^(eth|ens)" || echo "No eth/ens interfaces found"

echo ""
echo "=== Recommendations ==="
echo "For modern NCCL (>= 2.18) + aws-ofi-nccl (>= 1.7.0):"
echo "  Most NCCL_* and FI_* env vars are auto-detected."
echo "  Only set NCCL_DEBUG=INFO for troubleshooting."
