#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

configs=experiments/coverage_dlgn/configs
results=experiments/coverage_dlgn/results

run_one() {
    local gpu=$1
    local budget=$2
    local depth=$3
    local strategy=$4
    local seed=$5
    local stem="depth_cifar10_budget${budget}_depth${depth}_${strategy}_v3"
    local config="$configs/${stem}_seed0.json"
    local output="$results/${stem}_seed${seed}"
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

run_pair() {
    local gpu=$1
    local budget=$2
    local depth=$3
    local seed=$4
    run_one "$gpu" "$budget" "$depth" random "$seed"
    run_one "$gpu" "$budget" "$depth" semantic_balanced "$seed"
}

run_gpu0() {
    run_pair 0 512k 8 0
    run_pair 0 48k 8 0
    run_pair 0 512k 8 2
    run_pair 0 48k 8 2
    run_pair 0 512k 12 1
    run_pair 0 48k 12 1
}

run_gpu1() {
    run_pair 1 512k 12 0
    run_pair 1 48k 12 0
    run_pair 1 512k 12 2
    run_pair 1 48k 12 2
    run_pair 1 512k 8 1
    run_pair 1 48k 8 1
}

run_gpu0 &
pid0=$!
run_gpu1 &
pid1=$!
wait "$pid0"
wait "$pid1"
