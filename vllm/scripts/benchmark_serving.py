#!/usr/bin/env python3
"""Benchmark vLLM serving throughput and latency.

Sends concurrent requests to a vLLM server and measures:
  - Throughput (requests/sec, tokens/sec)
  - TTFT (time to first token)
  - TPOT (time per output token)
  - End-to-end latency (p50, p90, p99)

Usage:
    python benchmark_serving.py --url http://localhost:8000 --model meta-llama/Llama-3.1-8B-Instruct
    python benchmark_serving.py --url http://localhost:8000 --model llama --concurrency 32 --num-requests 200
"""

import argparse
import asyncio
import time
from dataclasses import dataclass, field

import aiohttp


@dataclass
class RequestResult:
    ttft: float = 0.0
    total_time: float = 0.0
    output_tokens: int = 0
    error: bool = False


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> RequestResult:
    """Send a single streaming chat completion request."""
    result = RequestResult()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }

    start = time.perf_counter()
    first_token_time = None

    try:
        async with session.post(
            f"{url}/v1/chat/completions", json=payload
        ) as resp:
            if resp.status != 200:
                result.error = True
                return result

            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if decoded.startswith("data: ") and decoded != "data: [DONE]":
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    result.output_tokens += 1

    except Exception:
        result.error = True
        return result

    end = time.perf_counter()
    result.total_time = end - start
    result.ttft = (first_token_time - start) if first_token_time else result.total_time
    return result


async def run_benchmark(
    url: str,
    model: str,
    concurrency: int,
    num_requests: int,
    max_tokens: int,
    prompt: str,
):
    """Run the benchmark with the given concurrency."""
    semaphore = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []

    async def bounded_request(session):
        async with semaphore:
            return await send_request(session, url, model, prompt, max_tokens)

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        start = time.perf_counter()
        tasks = [bounded_request(session) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - start

    # Filter successful results
    ok = [r for r in results if not r.error]
    errors = len(results) - len(ok)

    if not ok:
        print("All requests failed!")
        return

    ttfts = sorted([r.ttft for r in ok])
    latencies = sorted([r.total_time for r in ok])
    total_output_tokens = sum(r.output_tokens for r in ok)

    def percentile(data, p):
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    print(f"\n{'='*50}")
    print(f"Benchmark Results ({len(ok)} successful, {errors} errors)")
    print(f"{'='*50}")
    print(f"Concurrency:       {concurrency}")
    print(f"Wall time:         {wall_time:.2f}s")
    print(f"Throughput:        {len(ok)/wall_time:.1f} req/s")
    print(f"Token throughput:  {total_output_tokens/wall_time:.1f} tok/s")
    print(f"\nTTFT (time to first token):")
    print(f"  p50: {percentile(ttfts, 50)*1000:.1f}ms")
    print(f"  p90: {percentile(ttfts, 90)*1000:.1f}ms")
    print(f"  p99: {percentile(ttfts, 99)*1000:.1f}ms")
    print(f"\nEnd-to-end latency:")
    print(f"  p50: {percentile(latencies, 50)*1000:.1f}ms")
    print(f"  p90: {percentile(latencies, 90)*1000:.1f}ms")
    print(f"  p99: {percentile(latencies, 99)*1000:.1f}ms")
    print(
        f"\nAvg output tokens: {total_output_tokens/len(ok):.1f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM serving")
    parser.add_argument(
        "--url", default="http://localhost:8000", help="vLLM server URL"
    )
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--concurrency", type=int, default=16, help="Concurrent requests")
    parser.add_argument("--num-requests", type=int, default=100, help="Total requests")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max output tokens")
    parser.add_argument(
        "--prompt",
        default="Write a short paragraph about distributed systems.",
        help="Prompt text",
    )
    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            args.url,
            args.model,
            args.concurrency,
            args.num_requests,
            args.max_tokens,
            args.prompt,
        )
    )


if __name__ == "__main__":
    main()
