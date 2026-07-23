#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

configs=experiments/coverage_dlgn/configs
results=experiments/coverage_dlgn/results

run_one() {
    local gpu=$1
    local variant=$2
    local seed=$3
    local config="$configs/pilot_conv_cifar10_small_${variant}_seed0.json"
    local output="$results/pilot_conv_cifar10_small_${variant}_seed${seed}"
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

run_gpu0() {
    run_one 0 random 0
    run_one 0 semantic_channel_v4 0
    run_one 0 random 2
}

run_gpu1() {
    run_one 1 random 1
    run_one 1 semantic_channel_v4 1
    run_one 1 semantic_channel_v4 2
}

run_gpu0 &
pid0=$!
run_gpu1 &
pid1=$!
wait "$pid0"
wait "$pid1"
