---
name: multiturn-benchmark
description: >
  Benchmark KV cache compression strategies (BF16 vs FP8 vs TurboQuant 4-bit) on vLLM
  with multi-turn workloads on MiniMax-M2.7 / MI300X/MI355x. Finds scenarios where compression
  outperforms baseline by creating memory pressure. Key result: TurboQuant achieves 85%
  cache hit vs FP8's 27% under memory pressure, 2.6× faster TTFT.
  Usage: /multiturn-benchmark [baseline|fp8|tq4bit] [--num-clients N] [--num-rounds N]
allowed-tools: Bash, Read, Grep, Glob
---

# Multi-Turn KV Cache Compression Benchmarking

**Tags**: vllm, kv-cache, compression, turboquant, fp8, multi-turn, benchmark, prefix-caching
**Model**: MiniMax-M2.7
**Hardware**: MI300X/MI355X (ROCm)

## Overview

This skill covers benchmarking KV cache compression strategies (BF16, FP8, TurboQuant 4-bit) on vLLM with multi-turn workloads. The goal is to find scenarios where compression techniques outperform baseline by creating memory pressure.

## Key Concepts

### When KV Cache Compression Shines

- Compression benefits appear when **memory is the bottleneck**
- Need: `total_tokens > GPU_capacity / compression_ratio`
- Under light load, baseline wins (no memory pressure, compression has overhead)
- Under heavy load, compression wins (avoids cache eviction)

### Cache Hit Rate

- `cache_hit_rate = cached_tokens / prompt_tokens`
- High rate (90%+) = KV cache reused, fast TTFT
- Low rate (<20%) = cache evicted, must recompute, slow TTFT
- Per-round rate should increase in later rounds (history cached)

### Memory Capacity Calculation

```
KV cache per token = 2 (K+V) × num_kv_heads × head_dim × dtype_bytes × num_layers

For MiniMax-M2.7 with TP=2:
- Layers: 62, KV heads: 8 (4 per GPU), Head dim: 128
- BF16: 127 KB/token
- FP8 (2×): 63.5 KB/token
- 4-bit (4×): 31.75 KB/token

Capacity at 0.6 util on 288GB GPU (simulating 192GB):
- BF16: ~756k tokens
- FP8: ~1.5M tokens
- 4-bit: ~3M tokens
```

## Directory Structure

```
benchmarks/multi_turn_tq/
├── run_benchmark.sh          # Main wrapper script
├── bench_multiturn_enhanced.py  # Python benchmark with per-round metrics
├── compare_results.py        # Compare multiple result files
├── results/multiturn/        # JSON result files
├── logs/                     # Server and benchmark logs
├── BENCHMARK_REPORT.md       # Latest results report
└── SKILL_MULTITURN_BENCHMARK.md  # This file
```

## Key Scripts

### run_benchmark.sh

Launches vLLM server and runs benchmark. Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `GPU_MEMORY_UTIL` | Fraction of GPU memory to use | 0.9 |
| `MAX_MODEL_LEN` | Max context length | 8192 |
| `OUTPUT_TOKENS` | Max output tokens per round | 100 |
| `SUB_QUESTION_TOKENS` | Input tokens per follow-up round | 200 |
| `ATTENTION_BACKEND` | Attention backend (empty for auto) | - |
| `KV_SKIP_LAYERS` | Layers to skip for KV quantization | - |
| `VLLM_TQ_DECODE_V3` | Enable TurboQuant decode v3 | - |

Command line args: `--kv-cache-dtype`, `--tag`, `--num-clients`, `--num-rounds`, `--common-prefix`, `--prefix-tokens`, `--port`, `--skip-server`

### bench_multiturn_enhanced.py

Python benchmark that:

- Runs multi-turn conversations with round barrier mode
- Captures actual `cached_tokens` from server (requires `--enable-prompt-tokens-details`)
- Records per-round TTFT and cache hit rate
- Uses actual model responses in history (critical for prefix caching!)

## Important Fixes Applied

### 1. History Must Use Actual Response

**Bug**: Original code used placeholder `"[Response for round N]"` instead of actual model output.
**Impact**: Prefix caching fails because history doesn't match cached KV.
**Fix**: Store `result.generated_text` and use it in history.

### 2. TurboQuant Server Flags

```bash
VLLM_TQ_DECODE_V3=1           # Enable decode v3
KV_SKIP_LAYERS=""              # Pass --kv-cache-dtype-skip-layers ""
# Do NOT set ATTENTION_BACKEND - let TQ auto-select
```

### 3. TurboQuant Memory Overhead

Known bug: TQ uses ~22GB more than expected. Workaround: increase `GPU_MEMORY_UTIL` by ~0.08 for TQ runs.

## Parameter Tuning Guide

### To Create Memory Pressure

Increase total tokens until baseline starts evicting:

```
total_tokens = num_clients × tokens_per_client
tokens_per_client = common_prefix + unique_prefix + rounds × (output + input)
```

### To See Gradual Degradation

Find settings where:

- Round 0-2: All fit
- Round 3-5: FP8 starts evicting, TQ still fits
- Round 6+: FP8 heavily evicting, TQ starts evicting

### Example Configurations

**Light load (no pressure)**:

- 16 clients, 8k prefix, 5 rounds → All methods similar

**Medium load (BF16 evicts)**:

- 40 clients, 16k prefix, 10 rounds → BF16 degrades, FP8/TQ OK

**Heavy load (FP8 evicts)**:

- 40 clients, 32k prefix, 8 rounds, 2k input/round → FP8 degrades, TQ wins

## Reproduction Commands

### Quick 3-Way Comparison

```bash
cd benchmarks/multi_turn_tq

# Run all 3 in parallel on different GPU pairs
HIP_VISIBLE_DEVICES=4,5 GPU_MEMORY_UTIL=0.6 MAX_MODEL_LEN=80000 \
OUTPUT_TOKENS=200 SUB_QUESTION_TOKENS=2000 ATTENTION_BACKEND=ROCM_AITER_FA \
./run_benchmark.sh --kv-cache-dtype auto --tag test_baseline \
--num-clients 40 --num-rounds 8 --common-prefix 2000 --prefix-tokens 32000 --port 6789 &

HIP_VISIBLE_DEVICES=2,3 GPU_MEMORY_UTIL=0.6 MAX_MODEL_LEN=80000 \
OUTPUT_TOKENS=200 SUB_QUESTION_TOKENS=2000 ATTENTION_BACKEND=ROCM_AITER_FA \
./run_benchmark.sh --kv-cache-dtype fp8_e4m3 --tag test_fp8 \
--num-clients 40 --num-rounds 8 --common-prefix 2000 --prefix-tokens 32000 --port 6791 &

HIP_VISIBLE_DEVICES=6,7 GPU_MEMORY_UTIL=0.68 MAX_MODEL_LEN=80000 \
OUTPUT_TOKENS=200 SUB_QUESTION_TOKENS=2000 VLLM_TQ_DECODE_V3=1 KV_SKIP_LAYERS="" \
./run_benchmark.sh --kv-cache-dtype turboquant_4bit_nc --tag test_tq4bit \
--num-clients 40 --num-rounds 8 --common-prefix 2000 --prefix-tokens 32000 --port 6790 &

wait

# Compare
python compare_results.py results/multiturn/results_test_*.json
```

### Simulating Different GPU Sizes

```bash
# 288GB GPU at 0.6 util ≈ 173GB (simulates 192GB)
# 288GB GPU at 0.5 util ≈ 144GB (simulates 160GB)
# 288GB GPU at 0.4 util ≈ 115GB (simulates 128GB)
```

## Interpreting Results

### Good Result Pattern

```
Per-Round Cache Hit Rate:
Round | BF16  | FP8   | TQ
0     | 5.8%  | 5.8%  | 5.8%    <- All same (cold start)
1     | 10%   | 73%   | 94%     <- TQ >> FP8 >> BF16
2     | 7%    | 48%   | 94%     <- FP8 degrading, TQ stable
...
7     | 4%    | 13%   | 95%     <- TQ maintains high hit rate
```

### Warning Signs

- Cache hit rate same across all methods → not enough memory pressure
- TQ cache hit dropping → exceeded even TQ capacity
- FP8 similar to TQ → workload too light for 4-bit advantage

## Troubleshooting

### Server Fails to Start

- Check logs in `logs/server_*.log`
- Verify GPU memory available: `rocm-smi --showmeminfo vram`
- Kill stale processes: `pkill -9 -f "vllm serve"`

### Low Cache Hit Despite Multi-Turn

- Verify history uses actual model response (not placeholder)
- Check `--enable-prompt-tokens-details` flag on server
- Ensure `--enable-prefix-caching` is set

### OOM Errors

- Reduce `--num-clients` or `--prefix-tokens`
- Lower `GPU_MEMORY_UTIL`
- Check for zombie GPU processes: `rocm-smi --showpids`

## Files Reference

| File | Purpose |
|------|---------|
| `results_*.json` | Raw benchmark results with per-round metrics |
| `all_results.jsonl` | Appended results for tracking |
| `server_*.log` | vLLM server output, includes cache stats |
| `benchmark_*.log` | Benchmark script output |
