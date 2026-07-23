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

random=experiments/coverage_dlgn/configs/pilot_cifar10_random_v3_seed0.json
semantic=experiments/coverage_dlgn/configs/pilot_cifar10_semantic_balanced_v3_seed0.json
results=experiments/coverage_dlgn/results

run_gpu0() {
    run_one 0 "$random" 0 "$results/pilot_cifar10_random_v3_seed0"
    run_one 0 "$semantic" 0 "$results/pilot_cifar10_semantic_balanced_v3_seed0"
    run_one 0 "$random" 2 "$results/pilot_cifar10_random_v3_seed2"
}

run_gpu1() {
    run_one 1 "$random" 1 "$results/pilot_cifar10_random_v3_seed1"
    run_one 1 "$semantic" 1 "$results/pilot_cifar10_semantic_balanced_v3_seed1"
    run_one 1 "$semantic" 2 "$results/pilot_cifar10_semantic_balanced_v3_seed2"
}

run_gpu0 &
pid0=$!
run_gpu1 &
pid1=$!
wait "$pid0"
wait "$pid1"
