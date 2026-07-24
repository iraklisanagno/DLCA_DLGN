#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

results=experiments/coverage_dlgn/results
scope=${1:-all}

if [[ "$scope" != "small" && "$scope" != "medium" && "$scope" != "all" ]]; then
    echo "Usage: $0 [small|medium|all]" >&2
    exit 2
fi

evaluate_one() {
    local gpu=$1
    local scale=$2
    local variant=$3
    local seed=$4
    local run_dir="$results/pilot_conv_cifar10_paper_${scale}_${variant}_seed${seed}"
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

evaluate_scale() {
    local scale=$1
    (
        evaluate_one 0 "$scale" random 0
        evaluate_one 0 "$scale" semantic_channel_v4 1
        evaluate_one 0 "$scale" random 2
    ) &
    local pid0=$!
    (
        evaluate_one 1 "$scale" semantic_channel_v4 0
        evaluate_one 1 "$scale" random 1
        evaluate_one 1 "$scale" semantic_channel_v4 2
    ) &
    local pid1=$!
    wait "$pid0"
    wait "$pid1"
}

if [[ "$scope" == "small" || "$scope" == "all" ]]; then
    evaluate_scale small
fi
if [[ "$scope" == "medium" || "$scope" == "all" ]]; then
    evaluate_scale medium
fi
