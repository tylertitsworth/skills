#!/usr/bin/env python3
"""Benchmark GPU matmul throughput across dtypes and sizes.

Useful for validating GPU health, comparing dtype performance,
and establishing baselines before training runs.

Usage:
    python benchmark_matmul.py
    python benchmark_matmul.py --sizes 1024 2048 4096 8192 --dtype float16 bfloat16
"""

import argparse
import time

import torch


def benchmark_matmul(
    m: int, n: int, k: int, dtype: torch.dtype, warmup: int = 10, iters: int = 100
) -> dict:
    """Benchmark a single matmul configuration."""
    a = torch.randn(m, k, dtype=dtype, device="cuda")
    b = torch.randn(k, n, dtype=dtype, device="cuda")

    # Warmup
    for _ in range(warmup):
        torch.mm(a, b)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Calculate TFLOPS: 2*M*N*K ops per matmul
    flops = 2 * m * n * k * iters
    tflops = flops / elapsed / 1e12

    return {
        "shape": f"{m}x{k} @ {k}x{n}",
        "dtype": str(dtype).split(".")[-1],
        "time_ms": elapsed / iters * 1000,
        "tflops": tflops,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU matmul throughput")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[1024, 2048, 4096, 8192],
        help="Square matrix sizes to benchmark",
    )
    parser.add_argument(
        "--dtype",
        nargs="+",
        default=["float32", "float16", "bfloat16"],
        help="Data types to benchmark",
    )
    parser.add_argument("--iters", type=int, default=100, help="Iterations per config")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available")
        return

    device = torch.cuda.get_device_name(0)
    print(f"GPU: {device}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    print(f"{'Shape':<22} {'Dtype':<10} {'Time (ms)':<12} {'TFLOPS':<10}")
    print("-" * 56)

    for size in args.sizes:
        for dt_name in args.dtype:
            dt = dtype_map.get(dt_name)
            if dt is None:
                print(f"Unknown dtype: {dt_name}")
                continue
            result = benchmark_matmul(size, size, size, dt, iters=args.iters)
            print(
                f"{result['shape']:<22} {result['dtype']:<10} "
                f"{result['time_ms']:<12.3f} {result['tflops']:<10.2f}"
            )


if __name__ == "__main__":
    main()
