#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

run_one() {
    local gpu=$1
    local config=$2
    local seed=$3
    local output=$4
    if [[ -e "$output" ]]; then
        echo "Refusing to append to existing output: $output" >&2
        return 1
    fi
    mkdir -p "$output"
    env DATASET_PATH=/tmp/torchlogix-datasets CUDA_VISIBLE_DEVICES="$gpu" \
        venv/bin/python experiments/train.py \
        --config "$config" --seed "$seed" --topology-seed "$seed" \
        --output "$output" > "$output/console.log" 2>&1
}

cifar_random=experiments/coverage_dlgn/configs/paper_cifar10_small_random_seed0.json
cifar_v3=experiments/coverage_dlgn/configs/paper_cifar10_small_semantic_balanced_v3_seed0.json
fashion_random=experiments/coverage_dlgn/configs/paper_fashion_mnist_small_random_seed0.json
fashion_v3=experiments/coverage_dlgn/configs/paper_fashion_mnist_small_semantic_balanced_v3_seed0.json
results=experiments/coverage_dlgn/results

run_seed() {
    local gpu=$1
    local seed=$2
    run_one "$gpu" "$cifar_random" "$seed" \
        "$results/paper_cifar10_small_random_seed${seed}"
    run_one "$gpu" "$cifar_v3" "$seed" \
        "$results/paper_cifar10_small_semantic_balanced_v3_seed${seed}"
    run_one "$gpu" "$fashion_random" "$seed" \
        "$results/paper_fashion_mnist_small_random_seed${seed}"
    run_one "$gpu" "$fashion_v3" "$seed" \
        "$results/paper_fashion_mnist_small_semantic_balanced_v3_seed${seed}"
}

run_seed 0 3 &
pid0=$!
run_seed 1 4 &
pid1=$!
wait "$pid0"
wait "$pid1"
