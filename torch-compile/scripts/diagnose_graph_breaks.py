#!/usr/bin/env python3
"""Diagnose torch.compile graph breaks in a model.

Wraps a model with fullgraph=True to find breaks, then reports them.
Also benchmarks compiled vs eager performance.

Usage:
    python diagnose_graph_breaks.py
    # Or import and use with your own model:
    # from diagnose_graph_breaks import diagnose_model
    # diagnose_model(my_model, sample_input)
"""

import time

import torch
import torch.nn as nn


def diagnose_model(model: nn.Module, sample_input: torch.Tensor, device: str = "cuda"):
    """Run graph break diagnosis on a model."""
    model = model.to(device)
    sample_input = sample_input.to(device)

    print("=== Step 1: Check for graph breaks ===\n")
    import torch._dynamo

    # Capture graph breaks
    torch._dynamo.config.verbose = True
    explanation = torch._dynamo.explain(model)(sample_input)

    print(f"\nGraph breaks: {explanation.break_count}")
    print(f"Compiled regions: {explanation.graph_count}")

    if explanation.break_count > 0:
        print("\nBreak reasons:")
        for i, reason in enumerate(explanation.break_reasons):
            print(f"  {i+1}. {reason}")

    # Reset for benchmarking
    torch._dynamo.reset()

    print("\n=== Step 2: Benchmark ===\n")

    # Eager baseline
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            model(sample_input)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(50):
            model(sample_input)
        torch.cuda.synchronize()
        eager_time = (time.perf_counter() - start) / 50 * 1000

    # Compiled
    for mode in ["default", "reduce-overhead", "max-autotune"]:
        torch._dynamo.reset()
        try:
            compiled = torch.compile(model, mode=mode)
            with torch.no_grad():
                # Warmup (includes compilation)
                for _ in range(3):
                    compiled(sample_input)
                torch.cuda.synchronize()

                start = time.perf_counter()
                for _ in range(50):
                    compiled(sample_input)
                torch.cuda.synchronize()
                compiled_time = (time.perf_counter() - start) / 50 * 1000

            speedup = eager_time / compiled_time
            print(
                f"  {mode:<30} {compiled_time:.2f}ms  "
                f"({speedup:.2f}x {'faster' if speedup > 1 else 'slower'})"
            )
        except Exception as e:
            print(f"  {mode:<30} FAILED: {e}")

    print(f"\n  {'eager (baseline)':<30} {eager_time:.2f}ms")


def main():
    """Demo with a simple transformer model."""
    if not torch.cuda.is_available():
        print("No CUDA GPU — running on CPU (results won't be representative)")
        device = "cpu"
    else:
        device = "cuda"
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # Simple model for demonstration
    model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
        num_layers=6,
    )

    sample_input = torch.randn(4, 128, 512)
    diagnose_model(model, sample_input, device)


if __name__ == "__main__":
    main()
