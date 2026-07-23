#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

while [[ ! -f experiments/coverage_dlgn/results/paper_fashion_mnist_small_random_seed1/run_summary.json ]]; do
    sleep 30
done

for seed in 1 2; do
    output="experiments/coverage_dlgn/results/paper_fashion_mnist_small_hybrid_v2_seed${seed}"
    if [[ -e "$output" ]]; then
        echo "Refusing to append to existing output: $output" >&2
        exit 1
    fi
    mkdir -p "$output"
    env DATASET_PATH=/tmp/torchlogix-datasets CUDA_VISIBLE_DEVICES=1 \
        venv/bin/python experiments/train.py \
        --config experiments/coverage_dlgn/configs/paper_fashion_mnist_small_hybrid_v2_seed0.json \
        --seed "$seed" --topology-seed "$seed" --output "$output" \
        > "$output/console.log" 2>&1
done

output=experiments/coverage_dlgn/results/paper_fashion_mnist_small_hybrid_f100_v2_seed0
if [[ -e "$output" ]]; then
    echo "Refusing to append to existing output: $output" >&2
    exit 1
fi
mkdir -p "$output"
env DATASET_PATH=/tmp/torchlogix-datasets CUDA_VISIBLE_DEVICES=1 \
    venv/bin/python experiments/train.py \
    --config experiments/coverage_dlgn/configs/paper_fashion_mnist_small_hybrid_v2_seed0.json \
    --coverage-long-range-fraction 1.0 --output "$output" \
    > "$output/console.log" 2>&1
