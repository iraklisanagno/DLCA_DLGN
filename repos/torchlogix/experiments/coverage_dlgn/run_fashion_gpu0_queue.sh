#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

while [[ ! -f experiments/coverage_dlgn/results/paper_fashion_mnist_small_hybrid_v2_seed0/run_summary.json ]]; do
    sleep 30
done

output=experiments/coverage_dlgn/results/paper_fashion_mnist_small_random_seed2
if [[ -e "$output" ]]; then
    echo "Refusing to append to existing output: $output" >&2
    exit 1
fi
mkdir -p "$output"
env DATASET_PATH=/tmp/torchlogix-datasets CUDA_VISIBLE_DEVICES=0 \
    venv/bin/python experiments/train.py \
    --config experiments/coverage_dlgn/configs/paper_fashion_mnist_small_random_seed0.json \
    --seed 2 --topology-seed 2 --output "$output" \
    > "$output/console.log" 2>&1

for specification in "0.00 f000" "0.50 f050" "0.75 f075"; do
    read -r fraction label <<< "$specification"
    output="experiments/coverage_dlgn/results/paper_fashion_mnist_small_hybrid_${label}_v2_seed0"
    if [[ -e "$output" ]]; then
        echo "Refusing to append to existing output: $output" >&2
        exit 1
    fi
    mkdir -p "$output"
    env DATASET_PATH=/tmp/torchlogix-datasets CUDA_VISIBLE_DEVICES=0 \
        venv/bin/python experiments/train.py \
        --config experiments/coverage_dlgn/configs/paper_fashion_mnist_small_hybrid_v2_seed0.json \
        --coverage-long-range-fraction "$fraction" --output "$output" \
        > "$output/console.log" 2>&1
done
