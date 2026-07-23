#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

results=experiments/coverage_dlgn/results

evaluate_one() {
    local gpu=$1
    local variant=$2
    local seed=$3
    local run_dir="$results/pilot_conv_cifar10_small_${variant}_seed${seed}"
    if [[ -e "$run_dir/test_metrics.json" || -e "$run_dir/inference_benchmark.json" ]]; then
        echo "Refusing to overwrite an existing held-out result: $run_dir" >&2
        return 1
    fi
    env DATASET_PATH=/tmp/torchlogix-datasets CUDA_VISIBLE_DEVICES="$gpu" \
        venv/bin/python experiments/coverage_dlgn/evaluate_checkpoint.py \
        "$run_dir" --device cuda > "$run_dir/evaluation_console.log" 2>&1
    env DATASET_PATH=/tmp/torchlogix-datasets CUDA_VISIBLE_DEVICES="$gpu" \
        venv/bin/python experiments/coverage_dlgn/benchmark_checkpoint.py \
        "$run_dir" --device cuda > "$run_dir/benchmark_console.log" 2>&1
}

run_gpu0() {
    evaluate_one 0 random 0
    evaluate_one 0 semantic_channel_v4 0
    evaluate_one 0 random 2
}

run_gpu1() {
    evaluate_one 1 random 1
    evaluate_one 1 semantic_channel_v4 1
    evaluate_one 1 semantic_channel_v4 2
}

run_gpu0 &
pid0=$!
run_gpu1 &
pid1=$!
wait "$pid0"
wait "$pid1"
