#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Enhanced Multi-Turn Benchmark for KV Cache Compression Comparison

This script benchmarks multi-turn conversations with vLLM, capturing:
- Per-round TTFT (Time to First Token)
- Actual cached_tokens from server (not estimated)
- Cache hit rate per round and overall

Inspired by SGLang's bench_multiturn.py but works with vLLM's OpenAI API.

Usage:
    # Start server with --enable-prompt-tokens-details
    vllm serve MODEL --enable-prompt-tokens-details --enable-prefix-caching ...

    # Run benchmark
    python bench_multiturn_enhanced.py --num-clients 16 --num-rounds 5

    # Compare KV cache strategies
    python bench_multiturn_enhanced.py --tag baseline
    python bench_multiturn_enhanced.py --tag tq4bit
"""

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm


@dataclass
class RequestResult:
    """Result of a single request."""

    success: bool
    ttft: float = 0.0  # Time to first token (seconds)
    latency: float = 0.0  # Total latency (seconds)
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    generated_text: str = ""  # Actual model response for history
    error: str = ""


@dataclass
class RoundMetrics:
    """Metrics for a single round across all clients."""

    ttft: list[float] = field(default_factory=list)
    latency: list[float] = field(default_factory=list)
    prompt_tokens: list[int] = field(default_factory=list)
    cached_tokens: list[int] = field(default_factory=list)
    completion_tokens: list[int] = field(default_factory=list)

    @property
    def cache_hit_rate(self) -> float:
        total_prompt = sum(self.prompt_tokens)
        total_cached = sum(self.cached_tokens)
        return total_cached / total_prompt if total_prompt > 0 else 0.0

    @property
    def avg_ttft(self) -> float:
        return mean(self.ttft) if self.ttft else 0.0

    @property
    def avg_latency(self) -> float:
        return mean(self.latency) if self.latency else 0.0


def percentile(values: list[float], p: float) -> float:
    """Calculate percentile of a list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(p * len(sorted_vals))
    if idx >= len(sorted_vals):
        idx = len(sorted_vals) - 1
    return sorted_vals[idx]


async def send_chat_request(
    session: aiohttp.ClientSession,
    url: str,
    messages: list[dict],
    model: str,
    max_tokens: int,
    timeout: float = 300.0,
) -> RequestResult:
    """Send a chat completion request and measure timing."""

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},  # Key: get cached_tokens!
    }

    result = RequestResult(success=False)
    start_time = time.perf_counter()
    ttft_recorded = False
    generated_text = ""

    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            if response.status != 200:
                result.error = f"HTTP {response.status}: {await response.text()}"
                return result

            async for line in response.content:
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]  # Remove "data: " prefix
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Record TTFT on first content
                if not ttft_recorded:
                    choices = data.get("choices", [])
                    if choices and choices[0].get("delta", {}).get("content"):
                        result.ttft = time.perf_counter() - start_time
                        ttft_recorded = True

                # Capture content
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if delta.get("content"):
                        generated_text += delta["content"]

                # Capture usage info (comes in final chunk)
                usage = data.get("usage")
                if usage:
                    result.prompt_tokens = usage.get("prompt_tokens", 0)
                    result.completion_tokens = usage.get("completion_tokens", 0)

                    # Get cached_tokens from prompt_tokens_details
                    details = usage.get("prompt_tokens_details")
                    if details:
                        result.cached_tokens = details.get("cached_tokens", 0)

            result.latency = time.perf_counter() - start_time
            result.success = True
            result.generated_text = generated_text  # Store actual response

            # If TTFT wasn't recorded (single chunk), use latency
            if not ttft_recorded:
                result.ttft = result.latency

    except asyncio.TimeoutError:
        result.error = "Request timeout"
    except Exception as e:
        result.error = str(e)

    return result


async def run_round(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    client_histories: list[list[dict]],
    max_tokens: int,
    max_parallel: int,
    pbar: tqdm,
) -> list[RequestResult]:
    """Run one round of requests for all clients."""

    semaphore = asyncio.Semaphore(max_parallel)

    async def send_one(messages: list[dict]) -> RequestResult:
        async with semaphore:
            result = await send_chat_request(session, url, messages, model, max_tokens)
            pbar.update(1)
            return result

    tasks = [send_one(history) for history in client_histories]
    return await asyncio.gather(*tasks)


def generate_text_tokens(tokenizer_name: str, num_tokens: int, seed: int = 42) -> str:
    """Generate random text of approximately num_tokens length."""
    # Simple word-based generation (approximately 1.3 tokens per word for English)
    random.seed(seed)
    words = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "hello",
        "world",
        "this",
        "is",
        "a",
        "test",
        "message",
        "for",
        "benchmarking",
        "language",
        "model",
        "performance",
        "with",
        "cache",
        "optimization",
        "and",
        "compression",
        "techniques",
        "that",
        "improve",
        "throughput",
        "latency",
        "memory",
        "efficiency",
        "in",
        "production",
        "systems",
        "running",
        "large",
        "scale",
        "inference",
        "workloads",
        "artificial",
        "intelligence",
        "machine",
        "learning",
        "deep",
        "neural",
        "networks",
        "transformers",
        "attention",
        "mechanism",
        "embeddings",
    ]

    # Approximate tokens needed (assuming ~1.3 tokens per word)
    num_words = int(num_tokens / 1.3)
    text = " ".join(random.choices(words, k=num_words))
    return text


class MultiTurnBenchmark:
    """Multi-turn benchmark with round barrier mode."""

    def __init__(self, args):
        self.args = args
        self.url = f"http://{args.host}:{args.port}/v1/chat/completions"
        self.round_metrics: dict[int, RoundMetrics] = {}

        # Initialize client histories with initial prompts
        self.client_histories: list[list[dict]] = []
        for i in range(args.num_clients):
            # Generate unique initial prompt for each client
            seed = args.seed + i

            # Common prefix (shared across all clients for prefix caching)
            common_text = generate_text_tokens(
                args.model, args.common_prefix_tokens, seed=args.seed
            )

            # Per-client unique context
            unique_text = generate_text_tokens(
                args.model, args.prefix_tokens, seed=seed
            )

            # Initial user message
            initial_content = (
                f"{common_text}\n\n"
                f"Context for conversation {i}: {unique_text}\n\n"
                f"Please summarize the above context and answer any follow-up questions."
            )

            self.client_histories.append([{"role": "user", "content": initial_content}])

    async def run(self) -> dict[str, Any]:
        """Run the benchmark with round barrier mode."""

        total_requests = self.args.num_clients * self.args.num_rounds

        print("\nStarting Multi-Turn Benchmark")
        print(f"  Clients: {self.args.num_clients}")
        print(f"  Rounds: {self.args.num_rounds}")
        print(f"  Total requests: {total_requests}")
        print(f"  Common prefix tokens: {self.args.common_prefix_tokens}")
        print(f"  Per-client prefix tokens: {self.args.prefix_tokens}")
        print(f"  Max output tokens: {self.args.output_tokens}")
        print(f"  Max parallel: {self.args.max_parallel}")
        print()

        connector = aiohttp.TCPConnector(limit=self.args.max_parallel * 2)
        timeout = aiohttp.ClientTimeout(total=600)

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            pbar = tqdm(total=total_requests, desc="Requests")
            start_time = time.perf_counter()

            for round_num in range(self.args.num_rounds):
                self.round_metrics[round_num] = RoundMetrics()

                # Send all requests for this round
                results = await run_round(
                    session=session,
                    url=self.url,
                    model=self.args.model,
                    client_histories=self.client_histories,
                    max_tokens=self.args.output_tokens,
                    max_parallel=self.args.max_parallel,
                    pbar=pbar,
                )

                # Process results and update histories
                for i, result in enumerate(results):
                    if not result.success:
                        print(f"Round {round_num}, Client {i} failed: {result.error}")
                        continue

                    # Record metrics
                    metrics = self.round_metrics[round_num]
                    metrics.ttft.append(result.ttft)
                    metrics.latency.append(result.latency)
                    metrics.prompt_tokens.append(result.prompt_tokens)
                    metrics.cached_tokens.append(result.cached_tokens)
                    metrics.completion_tokens.append(result.completion_tokens)

                    # Update history with actual assistant response
                    # This is critical for prefix caching to work correctly!
                    self.client_histories[i].append(
                        {"role": "assistant", "content": result.generated_text}
                    )

                    # Add next user message (sub-question)
                    if round_num < self.args.num_rounds - 1:
                        sub_question = generate_text_tokens(
                            self.args.model,
                            self.args.sub_question_tokens,
                            seed=self.args.seed + round_num * 1000 + i,
                        )
                        self.client_histories[i].append(
                            {
                                "role": "user",
                                "content": f"Follow-up question {round_num + 1}: {sub_question}",
                            }
                        )

                # Print round summary
                metrics = self.round_metrics[round_num]
                print(
                    f"\n  Round {round_num}: "
                    f"TTFT={metrics.avg_ttft:.3f}s, "
                    f"Cache Hit Rate={metrics.cache_hit_rate:.2%}, "
                    f"Cached={sum(metrics.cached_tokens)}/{sum(metrics.prompt_tokens)} tokens"
                )

            pbar.close()
            total_time = time.perf_counter() - start_time

        return self._generate_report(total_time)

    def _generate_report(self, total_time: float) -> dict[str, Any]:
        """Generate the final report."""

        # Aggregate all metrics
        all_ttft = []
        all_latency = []
        all_prompt_tokens = []
        all_cached_tokens = []
        all_completion_tokens = []

        for metrics in self.round_metrics.values():
            all_ttft.extend(metrics.ttft)
            all_latency.extend(metrics.latency)
            all_prompt_tokens.extend(metrics.prompt_tokens)
            all_cached_tokens.extend(metrics.cached_tokens)
            all_completion_tokens.extend(metrics.completion_tokens)

        total_prompt = sum(all_prompt_tokens)
        total_cached = sum(all_cached_tokens)
        total_completion = sum(all_completion_tokens)
        overall_cache_hit_rate = (
            total_cached / total_prompt if total_prompt > 0 else 0.0
        )

        report = {
            "config": {
                "num_clients": self.args.num_clients,
                "num_rounds": self.args.num_rounds,
                "common_prefix_tokens": self.args.common_prefix_tokens,
                "prefix_tokens": self.args.prefix_tokens,
                "sub_question_tokens": self.args.sub_question_tokens,
                "output_tokens": self.args.output_tokens,
                "model": self.args.model,
                "tag": self.args.tag,
            },
            "summary": {
                "total_requests": len(all_ttft),
                "total_time_sec": total_time,
                "throughput_req_per_sec": len(all_ttft) / total_time,
                "input_throughput_tok_per_sec": total_prompt / total_time,
                "output_throughput_tok_per_sec": total_completion / total_time,
                "overall_cache_hit_rate": overall_cache_hit_rate,
                "total_prompt_tokens": total_prompt,
                "total_cached_tokens": total_cached,
                "total_completion_tokens": total_completion,
                "ttft": {
                    "mean": mean(all_ttft) if all_ttft else 0,
                    "median": median(all_ttft) if all_ttft else 0,
                    "p90": percentile(all_ttft, 0.9),
                    "p99": percentile(all_ttft, 0.99),
                    "max": max(all_ttft) if all_ttft else 0,
                },
                "latency": {
                    "mean": mean(all_latency) if all_latency else 0,
                    "median": median(all_latency) if all_latency else 0,
                    "p90": percentile(all_latency, 0.9),
                    "p99": percentile(all_latency, 0.99),
                    "max": max(all_latency) if all_latency else 0,
                },
            },
            "per_round": {},
        }

        # Per-round breakdown
        for round_num, metrics in self.round_metrics.items():
            report["per_round"][f"round_{round_num}"] = {
                "num_requests": len(metrics.ttft),
                "cache_hit_rate": metrics.cache_hit_rate,
                "total_prompt_tokens": sum(metrics.prompt_tokens),
                "total_cached_tokens": sum(metrics.cached_tokens),
                "ttft_mean": metrics.avg_ttft,
                "ttft_p90": percentile(metrics.ttft, 0.9),
                "latency_mean": metrics.avg_latency,
            }

        return report


def print_report(report: dict[str, Any]):
    """Print a formatted report."""

    print("\n" + "=" * 70)
    print("MULTI-TURN BENCHMARK RESULTS")
    print("=" * 70)

    config = report["config"]
    print("\nConfiguration:")
    print(f"  Model: {config['model']}")
    print(f"  Tag: {config['tag']}")
    print(f"  Clients: {config['num_clients']}, Rounds: {config['num_rounds']}")
    print(f"  Common Prefix: {config['common_prefix_tokens']} tokens")
    print(f"  Per-Client Prefix: {config['prefix_tokens']} tokens")

    summary = report["summary"]
    print("\nOverall Summary:")
    print(f"  Total Requests: {summary['total_requests']}")
    print(f"  Total Time: {summary['total_time_sec']:.2f}s")
    print(f"  Throughput: {summary['throughput_req_per_sec']:.2f} req/s")
    print(f"  Input Throughput: {summary['input_throughput_tok_per_sec']:.0f} tok/s")
    print(f"  Output Throughput: {summary['output_throughput_tok_per_sec']:.0f} tok/s")

    print(f"\n  CACHE HIT RATE: {summary['overall_cache_hit_rate']:.2%}")
    print(
        f"    Cached: {summary['total_cached_tokens']:,} / {summary['total_prompt_tokens']:,} tokens"
    )

    ttft = summary["ttft"]
    print("\n  TTFT (Time to First Token):")
    print(
        f"    Mean: {ttft['mean'] * 1000:.1f}ms, Median: {ttft['median'] * 1000:.1f}ms"
    )
    print(f"    P90: {ttft['p90'] * 1000:.1f}ms, P99: {ttft['p99'] * 1000:.1f}ms")

    latency = summary["latency"]
    print("\n  Latency (End-to-End):")
    print(
        f"    Mean: {latency['mean'] * 1000:.1f}ms, Median: {latency['median'] * 1000:.1f}ms"
    )
    print(f"    P90: {latency['p90'] * 1000:.1f}ms, P99: {latency['p99'] * 1000:.1f}ms")

    print("\nPer-Round Breakdown:")
    print(f"  {'Round':<8} {'TTFT Mean':>12} {'Cache Hit':>12} {'Cached Tokens':>15}")
    print(f"  {'-' * 8} {'-' * 12} {'-' * 12} {'-' * 15}")

    for round_key in sorted(report["per_round"].keys()):
        r = report["per_round"][round_key]
        round_num = round_key.replace("round_", "")
        print(
            f"  {round_num:<8} "
            f"{r['ttft_mean'] * 1000:>10.1f}ms "
            f"{r['cache_hit_rate']:>11.1%} "
            f"{r['total_cached_tokens']:>7,}/{r['total_prompt_tokens']:,}"
        )

    print("=" * 70)


async def check_server(url: str) -> bool:
    """Check if server is available and supports required features."""
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get(
                url.replace("/v1/chat/completions", "/v1/models"),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response,
        ):
            return response.status == 200
    except Exception:
        return False


async def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-Turn Benchmark for KV Cache Compression"
    )

    # Server settings
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=6789)
    parser.add_argument(
        "--model",
        type=str,
        default="/shareddata/MiniMaxAI/MiniMax-M2.7",
        help="Model name (as registered in vLLM)",
    )

    # Benchmark settings
    parser.add_argument(
        "--num-clients",
        type=int,
        default=16,
        help="Number of concurrent clients/conversations",
    )
    parser.add_argument(
        "--num-rounds", type=int, default=5, help="Number of turns per conversation"
    )
    parser.add_argument(
        "--max-parallel", type=int, default=32, help="Maximum parallel requests"
    )

    # Token settings
    parser.add_argument(
        "--common-prefix-tokens",
        type=int,
        default=1000,
        help="Shared prefix tokens across all clients (for prefix caching)",
    )
    parser.add_argument(
        "--prefix-tokens",
        type=int,
        default=2000,
        help="Unique prefix tokens per client",
    )
    parser.add_argument(
        "--sub-question-tokens",
        type=int,
        default=200,
        help="Tokens per sub-question in subsequent rounds",
    )
    parser.add_argument(
        "--output-tokens", type=int, default=100, help="Max output tokens per response"
    )

    # Output settings
    parser.add_argument(
        "--tag", type=str, default="", help="Tag for this benchmark run"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/multiturn",
        help="Directory for results",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Check server
    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    print(f"Checking server at {url}...")
    if not await check_server(url):
        print(f"ERROR: Server not available at {url}")
        print("Make sure to start the server with:")
        print(
            "  vllm serve MODEL --enable-prompt-tokens-details --enable-prefix-caching ..."
        )
        return

    print("Server is available!")

    # Run benchmark
    benchmark = MultiTurnBenchmark(args)
    report = await benchmark.run()

    # Print results
    print_report(report)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or "default"
    output_file = output_dir / f"results_{tag}_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Also append to JSONL for easy comparison
    jsonl_file = output_dir / "all_results.jsonl"
    with open(jsonl_file, "a") as f:
        f.write(json.dumps({"timestamp": timestamp, "tag": tag, **report}) + "\n")

    print(f"Appended to: {jsonl_file}")


if __name__ == "__main__":
    asyncio.run(main())
