#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

fashion_runs=(
    paper_fashion_mnist_small_random_seed0
    paper_fashion_mnist_small_random_seed1
    paper_fashion_mnist_small_random_seed2
    paper_fashion_mnist_small_hybrid_v2_seed0
    paper_fashion_mnist_small_hybrid_v2_seed1
    paper_fashion_mnist_small_hybrid_v2_seed2
    paper_fashion_mnist_small_hybrid_f000_v2_seed0
    paper_fashion_mnist_small_hybrid_f050_v2_seed0
    paper_fashion_mnist_small_hybrid_f075_v2_seed0
    paper_fashion_mnist_small_hybrid_f100_v2_seed0
)
while true; do
    complete=true
    for run in "${fashion_runs[@]}"; do
        if [[ ! -f "experiments/coverage_dlgn/results/$run/run_summary.json" ]]; then
            complete=false
            break
        fi
    done
    "$complete" && break
    sleep 30
done

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

random_config=experiments/coverage_dlgn/configs/paper_cifar10_small_random_seed0.json
hybrid_config=experiments/coverage_dlgn/configs/paper_cifar10_small_hybrid_v2_seed0.json
results=experiments/coverage_dlgn/results

# Reproduce one complete published-architecture baseline before parallel work.
run_one 0 "$random_config" 0 "$results/paper_cifar10_small_random_seed0"

# Finish the paired random baselines before running the proposed topology.
run_one 0 "$random_config" 2 "$results/paper_cifar10_small_random_seed2" &
pid0=$!
run_one 1 "$random_config" 1 "$results/paper_cifar10_small_random_seed1" &
pid1=$!
wait "$pid0"
wait "$pid1"

run_one 0 "$hybrid_config" 0 "$results/paper_cifar10_small_hybrid_v2_seed0" &
pid0=$!
run_one 1 "$hybrid_config" 1 "$results/paper_cifar10_small_hybrid_v2_seed1" &
pid1=$!
wait "$pid0"
wait "$pid1"

run_one 0 "$hybrid_config" 2 "$results/paper_cifar10_small_hybrid_v2_seed2"
