#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

stage=${1:-screen}
configs=experiments/coverage_dlgn/configs/warp_fig4_cifar10_medium
results=experiments/coverage_dlgn/results

run_one() {
    local gpu=$1
    local arm=$2
    local seed=$3
    local config="$configs/${arm}_seed0.json"
    local output="$results/warp_fig4_medium_${arm}_seed${seed}"

    if [[ -f "$output/run_summary.json" || -f "$output/early_stop.json" ]]; then
        echo "SKIP complete: $output"
        return 0
    fi
    if [[ -e "$output" ]]; then
        echo "Refusing incomplete/existing output: $output" >&2
        return 1
    fi

    mkdir -p "$output"
    args=(
        --config "$config"
        --seed "$seed"
        --output "$output"
    )
    case "$arm" in
        paired_random_*|legacy_v4_*)
            args+=(--topology-seed "$seed")
            ;;
    esac

    echo "START gpu=$gpu arm=$arm seed=$seed"
    env DATASET_PATH=/tmp/torchlogix-datasets CUDA_VISIBLE_DEVICES="$gpu" \
        venv/bin/python experiments/train.py "${args[@]}" \
        > "$output/console.log" 2>&1
    echo "DONE gpu=$gpu arm=$arm seed=$seed"
}

screen_gpu0() {
    run_one 0 warp_fixed_uniform 0
    run_one 0 warp_learnable 0
    run_one 0 legacy_v4_fixed_uniform 0
}

screen_gpu1() {
    run_one 1 warp_fixed_distributive 0
    run_one 1 paired_random_fixed_uniform 0
}

fixed_full_gpu0() {
    run_one 0 warp_fixed_uniform 1
    run_one 0 warp_fixed_distributive 1
    run_one 0 paired_random_fixed_uniform 1
    run_one 0 legacy_v4_fixed_uniform 1
}

fixed_full_gpu1() {
    run_one 1 warp_fixed_uniform 2
    run_one 1 warp_fixed_distributive 2
    run_one 1 paired_random_fixed_uniform 2
    run_one 1 legacy_v4_fixed_uniform 2
}

learnable_full_gpu0() {
    run_one 0 warp_learnable 1
    run_one 0 paired_random_learnable 0
    run_one 0 legacy_v4_learnable 0
    run_one 0 paired_random_learnable 2
    run_one 0 legacy_v4_learnable 2
}

learnable_full_gpu1() {
    run_one 1 warp_learnable 2
    run_one 1 paired_random_learnable 1
    run_one 1 legacy_v4_learnable 1
}

case "$stage" in
    screen)
        screen_gpu0 &
        pid0=$!
        screen_gpu1 &
        pid1=$!
        wait "$pid0"
        wait "$pid1"
        ;;
    fixed-full)
        fixed_full_gpu0 &
        pid0=$!
        fixed_full_gpu1 &
        pid1=$!
        wait "$pid0"
        wait "$pid1"
        ;;
    learnable-full)
        learnable_full_gpu0 &
        pid0=$!
        learnable_full_gpu1 &
        pid1=$!
        wait "$pid0"
        wait "$pid1"
        ;;
    *)
        echo "Usage: $0 {screen|fixed-full|learnable-full}" >&2
        exit 2
        ;;
esac
