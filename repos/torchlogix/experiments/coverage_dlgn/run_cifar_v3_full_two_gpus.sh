#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

run_one() {
    local gpu=$1
    local seed=$2
    local output="experiments/coverage_dlgn/results/paper_cifar10_small_semantic_balanced_v3_seed${seed}"
    if [[ -e "$output" ]]; then
        echo "Refusing to append to existing output: $output" >&2
        return 1
    fi
    mkdir -p "$output"
    env DATASET_PATH=/tmp/torchlogix-datasets CUDA_VISIBLE_DEVICES="$gpu" \
        venv/bin/python experiments/train.py \
        --config experiments/coverage_dlgn/configs/paper_cifar10_small_semantic_balanced_v3_seed0.json \
        --seed "$seed" --topology-seed "$seed" --output "$output" \
        > "$output/console.log" 2>&1
}

run_one 0 0 &
pid0=$!
run_one 1 1 &
pid1=$!
wait "$pid0"
wait "$pid1"
run_one 0 2
