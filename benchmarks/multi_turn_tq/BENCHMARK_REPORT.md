# Multi-Turn KV Cache Compression Benchmark Report

## Summary

Comparing KV cache compression strategies on MiniMax-M2.7 with TP=2, simulating 192GB GPU memory.

**TurboQuant 4-bit achieves 85.7% cache hit rate vs FP8's 27.5%**, resulting in:

- **4.6× faster TTFT** than BF16 baseline (8.0s vs 37s)
- **2.4× faster TTFT** than FP8 (8.0s vs 19.5s)
- **1.3× faster total duration** than FP8 (492s vs 639s)

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | MiniMax-M2.7 |
| TP Size | 2 |
| GPU Memory Util | 0.6 |
| Clients | 40 |
| Rounds | 8 |
| Common Prefix | 2,000 tokens |
| Per-Client Prefix | 32,000 tokens |
| Input/Round | 2,000 tokens |
| Output/Round | 200 tokens |

## Results

### Overall Metrics

| Metric | BF16 | FP8 | TQ 4-bit |
|--------|------|-----|----------|
| TTFT Mean | 36,986 ms | 19,497 ms | **8,006 ms** |
| TTFT P90 | 63,394 ms | 43,452 ms | **10,580 ms** |
| Cache Hit Rate | 6.6% | 27.5% | **85.7%** |
| Throughput | 12,154 tok/s | 16,159 tok/s | **21,038 tok/s** |
| Total Duration | 849s | 639s | **492s** |

### Per-Round Cache Hit Rate

| Round | BF16 | FP8 | TQ 4-bit |
|-------|------|-----|----------|
| 0 | 5.8% | 5.8% | 5.8% |
| 1 | 10.0% | 73.5% | **93.7%** |
| 2 | 7.5% | 48.4% | **94.1%** |
| 3 | 10.7% | 31.3% | **94.4%** |
| 4 | 7.0% | 22.4% | **94.7%** |
| 5 | 4.5% | 18.5% | **95.0%** |
| 6 | 4.3% | 15.6% | **95.2%** |
| 7 | 4.1% | 13.2% | **95.4%** |

### Per-Round TTFT (ms)

| Round | BF16 | FP8 | TQ 4-bit |
|-------|------|-----|----------|
| 0 | 27,723 | 21,111 | 29,844 |
| 1 | 28,104 | 3,676 | **4,289** |
| 2 | 31,519 | 11,413 | **4,382** |
| 3 | 33,229 | 16,977 | **4,701** |
| 4 | 37,693 | 20,896 | **4,903** |
| 5 | 42,181 | 23,893 | **5,002** |
| 6 | 45,708 | 26,919 | **5,295** |
| 7 | 49,731 | 31,086 | **5,628** |

## Reproduction Steps

```bash
cd benchmarks/multi_turn_tq

# BF16 Baseline
HIP_VISIBLE_DEVICES=4,5 \
GPU_MEMORY_UTIL=0.6 \
MAX_MODEL_LEN=80000 \
OUTPUT_TOKENS=200 \
SUB_QUESTION_TOKENS=2000 \
ATTENTION_BACKEND=ROCM_AITER_FA \
./run_benchmark.sh \
    --kv-cache-dtype auto \
    --tag fix_baseline \
    --num-clients 40 \
    --num-rounds 8 \
    --common-prefix 2000 \
    --prefix-tokens 32000 \
    --port 6789

# FP8
HIP_VISIBLE_DEVICES=2,3 \
GPU_MEMORY_UTIL=0.6 \
MAX_MODEL_LEN=80000 \
OUTPUT_TOKENS=200 \
SUB_QUESTION_TOKENS=2000 \
ATTENTION_BACKEND=ROCM_AITER_FA \
./run_benchmark.sh \
    --kv-cache-dtype fp8_e4m3 \
    --tag fix_fp8 \
    --num-clients 40 \
    --num-rounds 8 \
    --common-prefix 2000 \
    --prefix-tokens 32000 \
    --port 6791

# TurboQuant 4-bit
HIP_VISIBLE_DEVICES=6,7 \
GPU_MEMORY_UTIL=0.6 \
MAX_MODEL_LEN=80000 \
OUTPUT_TOKENS=200 \
SUB_QUESTION_TOKENS=2000 \
VLLM_TQ_DECODE_V3=1 \
./run_benchmark.sh \
    --kv-cache-dtype turboquant_4bit_nc \
    --tag fix3_tq4bit \
    --num-clients 40 \
    --num-rounds 8 \
    --common-prefix 2000 \
    --prefix-tokens 32000 \
    --port 6790

# Compare results
python compare_results.py results/multiturn/results_fix_baseline_*.json results/multiturn/results_fix_fp8_*.json results/multiturn/results_fix3_tq4bit_*.json
```

## Result Files

- `results/multiturn/results_fix_baseline_20260429_022339.json`
- `results/multiturn/results_fix_fp8_20260429_022010.json`
- `results/multiturn/results_fix3_tq4bit_20260429_202437.json`
