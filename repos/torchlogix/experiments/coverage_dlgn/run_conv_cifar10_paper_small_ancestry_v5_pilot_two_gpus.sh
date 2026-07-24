#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

configs=experiments/coverage_dlgn/configs
results=experiments/coverage_dlgn/results

run_one() {
    local gpu=$1
    local variant=$2
    local seed=$3
    local config="$configs/pilot_conv_cifar10_paper_small_${variant}_seed${seed}.json"
    local output="$results/pilot_conv_cifar10_paper_small_${variant}_seed${seed}"
    if [[ -e "$output" ]]; then
        echo "Refusing to append to existing output: $output" >&2
        return 1
    fi
    env DATASET_PATH=/tmp/torchlogix-datasets CUDA_VISIBLE_DEVICES="$gpu" \
        venv/bin/python experiments/train.py --config "$config"
}

# Run each seed as a simultaneous pair. A later seed begins only after both
# variants in the current pair finish successfully.
for seed in 0 1 2; do
    run_one 0 random_controlled "$seed" &
    random_pid=$!
    run_one 1 ancestry_v5 "$seed" &
    ancestry_pid=$!
    wait "$random_pid"
    wait "$ancestry_pid"
done
