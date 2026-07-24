#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

configs=experiments/coverage_dlgn/configs
results=experiments/coverage_dlgn/results
scope=${1:-all}

if [[ "$scope" != "small" && "$scope" != "medium" && "$scope" != "all" ]]; then
    echo "Usage: $0 [small|medium|all]" >&2
    exit 2
fi

run_one() {
    local gpu=$1
    local scale=$2
    local variant=$3
    local seed=$4
    local config="$configs/pilot_conv_cifar10_paper_${scale}_${variant}_seed0.json"
    local output="$results/pilot_conv_cifar10_paper_${scale}_${variant}_seed${seed}"
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

# The S paired pilot is completed before M.  Each GPU alternates random and v4
# runs so method and GPU are not confounded.
run_small_gpu0() {
    run_one 0 small random 0
    run_one 0 small semantic_channel_v4 1
    run_one 0 small random 2
}

run_small_gpu1() {
    run_one 1 small semantic_channel_v4 0
    run_one 1 small random 1
    run_one 1 small semantic_channel_v4 2
}

if [[ "$scope" == "small" || "$scope" == "all" ]]; then
    run_small_gpu0 &
    small_pid0=$!
    run_small_gpu1 &
    small_pid1=$!
    wait "$small_pid0"
    wait "$small_pid1"
fi

run_medium_gpu0() {
    run_one 0 medium random 0
    run_one 0 medium semantic_channel_v4 1
    run_one 0 medium random 2
}

run_medium_gpu1() {
    run_one 1 medium semantic_channel_v4 0
    run_one 1 medium random 1
    run_one 1 medium semantic_channel_v4 2
}

if [[ "$scope" == "medium" || "$scope" == "all" ]]; then
    run_medium_gpu0 &
    medium_pid0=$!
    run_medium_gpu1 &
    medium_pid1=$!
    wait "$medium_pid0"
    wait "$medium_pid1"
fi
