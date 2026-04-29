#!/usr/bin/env bash
#
# Run Enhanced Multi-Turn Benchmark for KV Cache Compression Comparison
#
# This script:
# 1. Starts vLLM server with proper flags (--enable-prompt-tokens-details)
# 2. Runs the enhanced benchmark that captures per-round TTFT and cache hit rate
# 3. Saves results for comparison
#
# Usage:
#   ./run_benchmark.sh                                    # Defaults
#   ./run_benchmark.sh --kv-cache-dtype turboquant_4bit_nc --tag tq4bit
#   ./run_benchmark.sh --kv-cache-dtype auto --tag baseline
#   ./run_benchmark.sh --skip-server --tag tq4bit         # Use existing server
#
# Quick comparison:
#   ./run_benchmark.sh --kv-cache-dtype auto --tag baseline
#   ./run_benchmark.sh --kv-cache-dtype turboquant_4bit_nc --tag tq4bit
#   python compare_results.py results/multiturn/results_*.json

set -e

# ============================================================================
# CONFIGURABLE PARAMETERS
# ============================================================================

# Model Configuration
MODEL="${MODEL:-/shareddata/MiniMaxAI/MiniMax-M2.7}"
TP_SIZE="${TP_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.9}"
PORT="${PORT:-6789}"

# KV Cache Configuration
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

# Attention Backend (leave empty to let vLLM auto-select)
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"

# Benchmark Configuration
NUM_CLIENTS="${NUM_CLIENTS:-16}"
NUM_ROUNDS="${NUM_ROUNDS:-5}"
MAX_PARALLEL="${MAX_PARALLEL:-32}"

# Token Configuration
COMMON_PREFIX_TOKENS="${COMMON_PREFIX_TOKENS:-1000}"
PREFIX_TOKENS="${PREFIX_TOKENS:-2000}"
SUB_QUESTION_TOKENS="${SUB_QUESTION_TOKENS:-200}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-100}"

# Result tagging
TAG="${TAG:-}"

# Control flags
SKIP_SERVER="${SKIP_SERVER:-0}"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results/multiturn"
LOGS_DIR="${SCRIPT_DIR}/logs"

# ============================================================================
# PARSE COMMAND LINE ARGUMENTS
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --kv-cache-dtype)
            KV_CACHE_DTYPE="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --num-clients)
            NUM_CLIENTS="$2"
            shift 2
            ;;
        --num-rounds)
            NUM_ROUNDS="$2"
            shift 2
            ;;
        --common-prefix)
            COMMON_PREFIX_TOKENS="$2"
            shift 2
            ;;
        --prefix-tokens)
            PREFIX_TOKENS="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --skip-server)
            SKIP_SERVER=1
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "KV Cache Options:"
            echo "  --kv-cache-dtype TYPE   KV cache dtype (auto, fp8_e4m3, turboquant_4bit_nc)"
            echo "  --tag NAME              Tag for result files"
            echo ""
            echo "Benchmark Options:"
            echo "  --num-clients N         Number of concurrent clients (default: $NUM_CLIENTS)"
            echo "  --num-rounds N          Number of conversation rounds (default: $NUM_ROUNDS)"
            echo "  --common-prefix N       Shared prefix tokens (default: $COMMON_PREFIX_TOKENS)"
            echo "  --prefix-tokens N       Per-client prefix tokens (default: $PREFIX_TOKENS)"
            echo ""
            echo "Server Options:"
            echo "  --port N                Server port (default: $PORT)"
            echo "  --skip-server           Use existing server (don't restart)"
            echo ""
            echo "Example - Compare KV cache strategies:"
            echo "  $0 --kv-cache-dtype auto --tag baseline"
            echo "  $0 --kv-cache-dtype turboquant_4bit_nc --tag tq4bit"
            echo "  python compare_results.py results/multiturn/results_*.json"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# SETUP
# ============================================================================

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# Generate tag from KV cache dtype if not provided
if [[ -z "$TAG" ]]; then
    TAG="${KV_CACHE_DTYPE//\//_}"
    TAG="${TAG//:/_}"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================"
echo "Enhanced Multi-Turn Benchmark"
echo "============================================"
echo "Model:            $MODEL"
echo "TP Size:          $TP_SIZE"
echo "KV Cache Dtype:   $KV_CACHE_DTYPE"
echo "Port:             $PORT"
echo ""
echo "Benchmark Settings:"
echo "  Num Clients:        $NUM_CLIENTS"
echo "  Num Rounds:         $NUM_ROUNDS"
echo "  Common Prefix:      $COMMON_PREFIX_TOKENS tokens"
echo "  Per-Client Prefix:  $PREFIX_TOKENS tokens"
echo "  Sub-Question:       $SUB_QUESTION_TOKENS tokens"
echo "  Output Tokens:      $OUTPUT_TOKENS"
echo ""
echo "Tag:              $TAG"
echo "============================================"

# ============================================================================
# SERVER MANAGEMENT
# ============================================================================

cleanup() {
    echo "Cleaning up..."
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
}

wait_for_server() {
    local port=$1
    local timeout=${2:-600}
    local pid=$3
    echo "Waiting for server on port $port (timeout: ${timeout}s)..."

    local start_time=$(date +%s)
    while true; do
        # Check if server process is still alive
        if [[ -n "$pid" ]] && ! kill -0 $pid 2>/dev/null; then
            echo "Server process died unexpectedly!"
            return 1
        fi

        if curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            echo "Server is ready!"
            return 0
        fi

        local elapsed=$(($(date +%s) - start_time))
        if [[ $elapsed -ge $timeout ]]; then
            echo "Timeout waiting for server"
            return 1
        fi

        sleep 5
    done
}

if [[ "$SKIP_SERVER" -eq 0 ]]; then
    trap cleanup EXIT

    # Kill any existing server on the port
    echo "Checking for existing server on port $PORT..."
    pkill -f "vllm.*--port.*$PORT" 2>/dev/null || true
    sleep 2

    # Build server command with --enable-prompt-tokens-details
    SERVER_CMD="vllm serve $MODEL \
        --tensor-parallel-size $TP_SIZE \
        --trust-remote-code \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEMORY_UTIL \
        --port $PORT \
        --kv-cache-dtype $KV_CACHE_DTYPE \
        --enable-prefix-caching \
        --enable-prompt-tokens-details \
        --enforce-eager"

    # Add attention backend only if specified
    if [[ -n "$ATTENTION_BACKEND" ]]; then
        SERVER_CMD="$SERVER_CMD --attention-backend $ATTENTION_BACKEND"
    fi

    # Add kv-cache-dtype-skip-layers if specified (even if empty string)
    if [[ -n "${KV_SKIP_LAYERS+x}" ]]; then
        SERVER_CMD="$SERVER_CMD --kv-cache-dtype-skip-layers \"$KV_SKIP_LAYERS\""
    fi

    echo ""
    echo "Starting vLLM server..."
    echo "Command: $SERVER_CMD"
    echo ""

    SERVER_LOG="${LOGS_DIR}/server_${TAG}_${TIMESTAMP}.log"
    $SERVER_CMD > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    echo "Server PID: $SERVER_PID"
    echo "Server log: $SERVER_LOG"

    if ! wait_for_server $PORT 600 $SERVER_PID; then
        echo "Failed to start server. Check log: $SERVER_LOG"
        tail -50 "$SERVER_LOG"
        exit 1
    fi
else
    echo "Using existing server on port $PORT"
fi

# ============================================================================
# RUN ENHANCED BENCHMARK
# ============================================================================

echo ""
echo "Running enhanced multi-turn benchmark..."
echo ""

BENCHMARK_LOG="${LOGS_DIR}/benchmark_${TAG}_${TIMESTAMP}.log"

python "${SCRIPT_DIR}/bench_multiturn_enhanced.py" \
    --host localhost \
    --port $PORT \
    --model "$MODEL" \
    --num-clients $NUM_CLIENTS \
    --num-rounds $NUM_ROUNDS \
    --max-parallel $MAX_PARALLEL \
    --common-prefix-tokens $COMMON_PREFIX_TOKENS \
    --prefix-tokens $PREFIX_TOKENS \
    --sub-question-tokens $SUB_QUESTION_TOKENS \
    --output-tokens $OUTPUT_TOKENS \
    --tag "$TAG" \
    --output-dir "$RESULTS_DIR" \
    2>&1 | tee "$BENCHMARK_LOG"

echo ""
echo "============================================"
echo "Benchmark Complete!"
echo "============================================"
echo "Results in: $RESULTS_DIR"
echo "Log: $BENCHMARK_LOG"
echo ""
echo "To compare results:"
echo "  python ${SCRIPT_DIR}/compare_results.py ${RESULTS_DIR}/results_*.json"
