#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

results=experiments/coverage_dlgn/results

evaluate_one() {
    local gpu=$1
    local run=$2
    local run_dir="$results/$run"
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
    evaluate_one 0 paper_cifar10_medium_semantic_balanced_v3_seed0
    evaluate_one 0 paper_cifar10_medium_random_seed0
    evaluate_one 0 paper_cifar10_medium_semantic_balanced_v3_seed2
    evaluate_one 0 paper_cifar10_medium_random_seed2
    evaluate_one 0 paper_cifar10_medium_semantic_balanced_v3_seed4
}

run_gpu1() {
    evaluate_one 1 paper_cifar10_medium_semantic_balanced_v3_seed1
    evaluate_one 1 paper_cifar10_medium_random_seed1
    evaluate_one 1 paper_cifar10_medium_semantic_balanced_v3_seed3
    evaluate_one 1 paper_cifar10_medium_random_seed3
    evaluate_one 1 paper_cifar10_medium_random_seed4
}

run_gpu0 &
pid0=$!
run_gpu1 &
pid1=$!
wait "$pid0"
wait "$pid1"
