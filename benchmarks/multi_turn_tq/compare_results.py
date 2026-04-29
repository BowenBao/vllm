#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Compare multi-turn benchmark results across different KV cache strategies.

Usage:
    python compare_results.py results/multiturn/results_*.json
    python compare_results.py results_baseline.json results_tq4bit.json results_fp8.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_result(filepath: str) -> dict[str, Any]:
    """Load a benchmark result JSON file."""
    with open(filepath) as f:
        return json.load(f)


def extract_tag(filepath: str) -> str:
    """Extract tag from filename like results_tagname_timestamp.json"""
    name = Path(filepath).stem
    parts = name.replace("results_", "").split("_")
    # Remove timestamp (last 2 parts: date_time)
    if len(parts) >= 3:
        return "_".join(parts[:-2])
    return name


def is_enhanced_format(data: dict) -> bool:
    """Check if this is from the enhanced benchmark."""
    return "per_round" in data


def safe_get(data: dict, *keys, default=None):
    """Safely get nested dict values."""
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return default
    return data


def format_value(val, fmt=".2f"):
    """Format a value, handling None."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:{fmt}}"
    return str(val)


def print_comparison_table(results: list[tuple[str, dict]]):
    """Print a comparison table of key metrics."""

    # Check if any results are from enhanced benchmark
    has_enhanced = any(is_enhanced_format(d) for _, d in results)

    if has_enhanced:
        # Enhanced format metrics (with cache hit rate!)
        metrics = [
            (
                "TTFT Mean (ms)",
                lambda d: safe_get(d, "summary", "ttft", "mean", default=0) * 1000
                if is_enhanced_format(d)
                else safe_get(d, "mean_ttft_ms"),
            ),
            (
                "TTFT P90 (ms)",
                lambda d: safe_get(d, "summary", "ttft", "p90", default=0) * 1000
                if is_enhanced_format(d)
                else safe_get(d, "p90_ttft_ms"),
            ),
            (
                "TTFT P99 (ms)",
                lambda d: safe_get(d, "summary", "ttft", "p99", default=0) * 1000
                if is_enhanced_format(d)
                else safe_get(d, "p99_ttft_ms"),
            ),
            (
                "Latency Mean (ms)",
                lambda d: safe_get(d, "summary", "latency", "mean", default=0) * 1000
                if is_enhanced_format(d)
                else safe_get(d, "mean_e2e_latency_ms"),
            ),
            (
                "Latency P99 (ms)",
                lambda d: safe_get(d, "summary", "latency", "p99", default=0) * 1000
                if is_enhanced_format(d)
                else safe_get(d, "p99_e2e_latency_ms"),
            ),
            (
                "Cache Hit Rate (%)",
                lambda d: safe_get(d, "summary", "overall_cache_hit_rate", default=0)
                * 100
                if is_enhanced_format(d)
                else None,
            ),
            (
                "Cached Tokens",
                lambda d: safe_get(d, "summary", "total_cached_tokens")
                if is_enhanced_format(d)
                else None,
            ),
            (
                "Prompt Tokens",
                lambda d: safe_get(d, "summary", "total_prompt_tokens")
                if is_enhanced_format(d)
                else None,
            ),
            (
                "Input Throughput (tok/s)",
                lambda d: safe_get(d, "summary", "input_throughput_tok_per_sec")
                if is_enhanced_format(d)
                else safe_get(d, "input_throughput"),
            ),
            (
                "Output Throughput (tok/s)",
                lambda d: safe_get(d, "summary", "output_throughput_tok_per_sec")
                if is_enhanced_format(d)
                else safe_get(d, "output_throughput"),
            ),
            (
                "Total Requests",
                lambda d: safe_get(d, "summary", "total_requests")
                if is_enhanced_format(d)
                else safe_get(d, "total_requests"),
            ),
            (
                "Total Duration (s)",
                lambda d: safe_get(d, "summary", "total_time_sec")
                if is_enhanced_format(d)
                else safe_get(d, "total_time_sec"),
            ),
        ]
    else:
        # Original vLLM format metrics
        metrics = [
            ("TTFT Mean (ms)", lambda d: safe_get(d, "mean_ttft_ms")),
            ("TTFT P50 (ms)", lambda d: safe_get(d, "median_ttft_ms")),
            ("TTFT P90 (ms)", lambda d: safe_get(d, "p90_ttft_ms")),
            ("TTFT P99 (ms)", lambda d: safe_get(d, "p99_ttft_ms")),
            ("TPOT Mean (ms)", lambda d: safe_get(d, "mean_tpot_ms")),
            ("TPOT P50 (ms)", lambda d: safe_get(d, "median_tpot_ms")),
            ("TPOT P99 (ms)", lambda d: safe_get(d, "p99_tpot_ms")),
            ("E2E Latency Mean (ms)", lambda d: safe_get(d, "mean_e2e_latency_ms")),
            ("E2E Latency P99 (ms)", lambda d: safe_get(d, "p99_e2e_latency_ms")),
            ("Input Throughput (tok/s)", lambda d: safe_get(d, "input_throughput")),
            ("Output Throughput (tok/s)", lambda d: safe_get(d, "output_throughput")),
            ("Total Requests", lambda d: safe_get(d, "total_requests")),
            ("Successful Requests", lambda d: safe_get(d, "successful_requests")),
            ("Failed Requests", lambda d: safe_get(d, "failed_requests")),
            ("Total Duration (s)", lambda d: safe_get(d, "total_time_sec")),
        ]

    # Calculate column widths
    tags = [tag for tag, _ in results]
    metric_width = max(len(m[0]) for m in metrics)
    col_widths = [max(12, len(tag) + 2) for tag in tags]

    # Print header
    header = f"{'Metric':<{metric_width}}"
    for tag, width in zip(tags, col_widths):
        header += f" | {tag:>{width}}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    # Print metrics
    for metric_name, extractor in metrics:
        row = f"{metric_name:<{metric_width}}"
        values = []
        for _, data in results:
            val = extractor(data)
            values.append(val)

        # Find best value (for latency metrics, lower is better)
        # For throughput, higher is better
        is_throughput = "throughput" in metric_name.lower()
        numeric_vals = [
            v for v in values if v is not None and isinstance(v, (int, float))
        ]

        best_val = None
        if numeric_vals:
            best_val = max(numeric_vals) if is_throughput else min(numeric_vals)

        for i, (val, width) in enumerate(zip(values, col_widths)):
            formatted = format_value(val)
            # Highlight best value
            if val is not None and val == best_val and len(numeric_vals) > 1:
                formatted = f"*{formatted}*"
            row += f" | {formatted:>{width}}"

        print(row)

    print("=" * len(header))
    print("* = best value")


def calculate_improvement(baseline: dict, comparison: dict) -> dict:
    """Calculate percentage improvement from baseline to comparison."""
    improvements = {}

    latency_metrics = [
        "mean_ttft_ms",
        "median_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "p99_tpot_ms",
        "mean_e2e_latency_ms",
    ]
    throughput_metrics = ["input_throughput", "output_throughput"]

    for metric in latency_metrics:
        base_val = safe_get(baseline, metric)
        comp_val = safe_get(comparison, metric)
        if base_val and comp_val and base_val > 0:
            # For latency, negative change is improvement
            pct = ((base_val - comp_val) / base_val) * 100
            improvements[metric] = pct

    for metric in throughput_metrics:
        base_val = safe_get(baseline, metric)
        comp_val = safe_get(comparison, metric)
        if base_val and comp_val and base_val > 0:
            # For throughput, positive change is improvement
            pct = ((comp_val - base_val) / base_val) * 100
            improvements[metric] = pct

    return improvements


def print_improvement_summary(results: list[tuple[str, dict]]):
    """Print improvement summary vs first result (baseline)."""
    if len(results) < 2:
        return

    baseline_tag, baseline_data = results[0]

    print(f"\n\nImprovement vs Baseline ({baseline_tag})")
    print("=" * 60)

    for tag, data in results[1:]:
        print(f"\n{tag}:")
        improvements = calculate_improvement(baseline_data, data)

        for metric, pct in improvements.items():
            direction = "better" if pct > 0 else "worse"
            sign = "+" if pct > 0 else ""
            print(f"  {metric}: {sign}{pct:.1f}% ({direction})")


def print_per_round_comparison(results: list[tuple[str, dict]]):
    """Print per-round TTFT and cache hit rate comparison."""

    # Filter to enhanced format only
    enhanced_results = [
        (tag, data) for tag, data in results if is_enhanced_format(data)
    ]
    if not enhanced_results:
        return

    print("\n\nPer-Round Comparison (TTFT and Cache Hit Rate)")
    print("=" * 80)

    # Get all round numbers
    all_rounds = set()
    for _, data in enhanced_results:
        all_rounds.update(data.get("per_round", {}).keys())
    rounds = sorted(all_rounds)

    # Print TTFT comparison
    print("\nTTFT Mean (ms) by Round:")
    header = f"{'Round':<10}"
    for tag, _ in enhanced_results:
        header += f" | {tag:>15}"
    print(header)
    print("-" * len(header))

    for round_key in rounds:
        round_num = round_key.replace("round_", "")
        row = f"{round_num:<10}"
        for _, data in enhanced_results:
            round_data = data.get("per_round", {}).get(round_key, {})
            ttft = round_data.get("ttft_mean", 0) * 1000
            row += f" | {ttft:>15.1f}"
        print(row)

    # Print Cache Hit Rate comparison
    print("\nCache Hit Rate (%) by Round:")
    header = f"{'Round':<10}"
    for tag, _ in enhanced_results:
        header += f" | {tag:>15}"
    print(header)
    print("-" * len(header))

    for round_key in rounds:
        round_num = round_key.replace("round_", "")
        row = f"{round_num:<10}"
        for _, data in enhanced_results:
            round_data = data.get("per_round", {}).get(round_key, {})
            hit_rate = round_data.get("cache_hit_rate", 0) * 100
            row += f" | {hit_rate:>14.1f}%"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Compare multi-turn benchmark results")
    parser.add_argument("files", nargs="+", help="Result JSON files to compare")
    parser.add_argument(
        "--baseline", help="Specify baseline file for improvement calculation"
    )
    parser.add_argument(
        "--per-round", action="store_true", help="Show per-round breakdown"
    )
    args = parser.parse_args()

    if not args.files:
        print("No result files provided")
        sys.exit(1)

    # Load all results
    results = []
    for filepath in args.files:
        try:
            data = load_result(filepath)
            tag = extract_tag(filepath)
            results.append((tag, data))
            print(f"Loaded: {filepath} (tag: {tag})")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    if not results:
        print("No valid results to compare")
        sys.exit(1)

    # Sort results (baseline first if specified)
    if args.baseline:
        baseline_tag = extract_tag(args.baseline)
        results.sort(key=lambda x: (x[0] != baseline_tag, x[0]))

    print("\n")
    print_comparison_table(results)
    print_improvement_summary(results)

    # Always show per-round for enhanced results
    if any(is_enhanced_format(d) for _, d in results):
        print_per_round_comparison(results)


if __name__ == "__main__":
    main()
