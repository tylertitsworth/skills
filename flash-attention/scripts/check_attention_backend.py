#!/usr/bin/env python3
"""Check which attention backends are available and benchmark them.

Reports:
  - Available SDPA backends for current GPU/dtype
  - flash-attn package version (if installed)
  - Benchmark comparison across backends

Usage:
    python check_attention_backend.py
    python check_attention_backend.py --seq-len 4096 --head-dim 128
"""

import argparse
import time

import torch
import torch.nn.functional as F


def check_backends():
    """Report available attention backends."""
    print("=== GPU Info ===")
    if not torch.cuda.is_available():
        print("No CUDA GPU available!")
        return
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    print("\n=== Flash Attention Package ===")
    try:
        import flash_attn
        print(f"flash-attn version: {flash_attn.__version__}")
    except ImportError:
        print("flash-attn: NOT INSTALLED")

    print("\n=== SDPA Backend Availability ===")
    from torch.nn.attention import SDPBackend, sdpa_kernel

    B, H, S, D = 1, 16, 256, 64
    backends = {
        "FLASH_ATTENTION": SDPBackend.FLASH_ATTENTION,
        "EFFICIENT_ATTENTION": SDPBackend.EFFICIENT_ATTENTION,
        "MATH": SDPBackend.MATH,
    }
    try:
        backends["CUDNN_ATTENTION"] = SDPBackend.CUDNN_ATTENTION
    except AttributeError:
        pass

    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        print(f"\n  dtype={dtype}:")
        q = torch.randn(B, H, S, D, dtype=dtype, device="cuda")
        k = torch.randn(B, H, S, D, dtype=dtype, device="cuda")
        v = torch.randn(B, H, S, D, dtype=dtype, device="cuda")

        for name, backend in backends.items():
            try:
                with sdpa_kernel(backend):
                    _ = F.scaled_dot_product_attention(q, k, v)
                print(f"    {name}: ✅ available")
            except RuntimeError as e:
                reason = str(e).split("\n")[0][:80]
                print(f"    {name}: ❌ {reason}")


def benchmark_backends(seq_len: int, head_dim: int, num_heads: int = 16):
    """Benchmark available backends."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    print(f"\n=== Benchmark (seq_len={seq_len}, head_dim={head_dim}, heads={num_heads}) ===")

    B = 4
    dtype = torch.bfloat16
    q = torch.randn(B, num_heads, seq_len, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(B, num_heads, seq_len, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(B, num_heads, seq_len, head_dim, dtype=dtype, device="cuda")

    warmup, iters = 10, 50
    backends = [
        ("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
        ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
        ("MATH", SDPBackend.MATH),
    ]

    for name, backend in backends:
        try:
            with sdpa_kernel(backend):
                for _ in range(warmup):
                    F.scaled_dot_product_attention(q, k, v, is_causal=True)
                torch.cuda.synchronize()

                start = time.perf_counter()
                for _ in range(iters):
                    F.scaled_dot_product_attention(q, k, v, is_causal=True)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) / iters * 1000

            print(f"  {name:<25} {elapsed:.2f} ms/iter")
        except RuntimeError:
            print(f"  {name:<25} N/A (unsupported)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=16)
    args = parser.parse_args()

    check_backends()
    benchmark_backends(args.seq_len, args.head_dim, args.num_heads)


if __name__ == "__main__":
    main()
