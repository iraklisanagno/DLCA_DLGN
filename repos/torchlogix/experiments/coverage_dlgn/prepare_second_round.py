#!/usr/bin/env python3
"""Generate immutable queues for the DATE second-round experiment plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
CONFIG_ROOT = ROOT / "configs" / "second_round"
QUEUE_ROOT = ROOT / "queues"
RESULT_ROOT = Path("experiments/coverage_dlgn/results")


def fixed_base(dataset: str, architecture: str, seed: int) -> dict:
    return {
        "dataset": dataset,
        "architecture": architecture,
        "device": "cuda",
        "seed": seed,
        "data_split_seed": 2027,
        "topology_seed": seed,
        "batch_size": 100,
        "num_iterations": 108_000,
        "eval_freq": 2_000,
        "valid_set_size": 0.1,
        "augmentation": "none",
        "learning_rate": 0.01,
        "binarization_num_batches": 1 if dataset == "mnist" else 100,
        "binarization": "fixed",
        "binarization_init": "uniform",
        "binarization_per": "global",
        "connections": "fixed",
        "connections_init_method": "random",
        "connections_gumbel": False,
        "lut_rank": 2,
        "parametrization": "raw",
        "forward_sampling": "soft",
        "weight_init": "random",
    }


def conv_base(seed: int) -> dict:
    return {
        "dataset": "cifar-10",
        "architecture": "ClgnCifar10PaperSmall",
        "device": "cuda",
        "seed": seed,
        "data_split_seed": 2027,
        "topology_seed": seed,
        "batch_size": 128,
        "num_iterations": 20_000,
        "eval_freq": 2_000,
        "valid_set_size": 0.1,
        "augmentation": "standard",
        "learning_rate": 0.02,
        "weight_decay": 0.002,
        "binarization_num_batches": 100,
        "binarization": "fixed",
        "binarization_init": "uniform",
        "binarization_per": "global",
        "connections": "fixed",
        "connections_init_method": "random",
        "conv_connections_init_method": "random",
        "classifier_connections_init_method": "random",
        "lut_rank": 2,
        "parametrization": "raw",
        "forward_sampling": "soft",
        "weight_init": "residual",
        "residual_probability": 0.951,
    }


def add_output(config: dict, name: str) -> dict:
    config = dict(config)
    config["output"] = str(RESULT_ROOT / name)
    return config


def exact_comparators() -> list[tuple[str, dict]]:
    entries = []
    specifications = {
        "mnist": {
            "dataset": "mnist",
            "prefix": "mnist",
            "mommen_candidates": 16,
            "lilogic_tau": 15.0,
            "fixed_arch": "DlgnMnistPaperSmallLearnable",
            "bit_arch": "DlgnMnistBitLogic48kDepth6",
        },
        "fashion": {
            "dataset": "fashion-mnist",
            "prefix": "fashion",
            "mommen_candidates": 8,
            "lilogic_tau": 25.0,
            "fixed_arch": "DlgnFashionMnistPaperSmallLearnable",
            "bit_arch": "DlgnFashionMnistBitLogic48kDepth6",
        },
    }
    for spec in specifications.values():
        for seed in range(3):
            mommen_name = (
                f"second_exact_{spec['prefix']}_mommen_6x8k_seed{seed}"
            )
            mommen = fixed_base(spec["dataset"], spec["fixed_arch"], seed)
            mommen.update({
                "connections_forward_mode": "soft_mix",
                "connections_num_candidates": spec["mommen_candidates"],
                "connections_temperature": 1.0,
                "connections_temperature_anneal_start": 2 / 3,
                "connections_temperature_anneal_end": 5 / 6,
                "connections_temperature_final": 1e-4,
                "connections_weights_init": "normal",
                "parametrization_temperature": 1.0,
                "parametrization_temperature_anneal_start": 5 / 6,
                "parametrization_temperature_anneal_end": 1.0,
                "parametrization_temperature_final": 1e-4,
            })
            entries.append((mommen_name, add_output(mommen, mommen_name)))

            lilogic_name = (
                f"second_exact_{spec['prefix']}_lilogic_6x8k_seed{seed}"
            )
            lilogic = fixed_base(spec["dataset"], spec["fixed_arch"], seed)
            lilogic.update({
                "batch_size": 256,
                "num_iterations": 42_200,
                "eval_freq": 844,
                "learning_rate": 0.075,
                "group_sum_temperature": spec["lilogic_tau"],
                "connections_forward_mode": "soft_mix",
                "connections_num_candidates": 32,
                "connections_temperature": 1.0,
                "connections_weights_init": "normal",
                "parametrization_temperature": 1.0,
            })
            entries.append((lilogic_name, add_output(lilogic, lilogic_name)))

            bit_name = (
                f"second_exact_{spec['prefix']}_bitlogic_6x8k_seed{seed}"
            )
            bit = fixed_base(spec["dataset"], spec["bit_arch"], seed)
            bit.update({
                "batch_size": 128,
                "num_iterations": 84_400,
                "eval_freq": 1_688,
                # Match the reproduced BitLogic protocol.  In particular,
                # MNIST must not inherit fixed_base()'s one-batch shortcut:
                # the distributive thresholds are calibrated on 100 batches.
                "binarization_num_batches": 100,
                "binarization_init": "distributive",
                "connections_forward_mode": "soft_mix",
                "connections_num_candidates": 16,
                "connections_temperature": 1.0,
                "connections_weights_init": "normal",
                "lut_rank": 4,
                "parametrization": "light",
                "parametrization_temperature": 1.0,
                "weight_decay": 0.0,
            })
            entries.append((bit_name, add_output(bit, bit_name)))
    return entries


def compression() -> list[tuple[str, dict]]:
    entries = []
    cells = [
        (
            "mnist",
            "mnist",
            {
                "4k": "DlgnMnistBudget4k",
                "8k": "DlgnMnistBudget8k",
                "16k": "DlgnMnistBudget16k",
                "32k": "DlgnMnistBudget32k",
            },
            0.25,
        ),
        (
            "fashion",
            "fashion-mnist",
            {
                "8k": "DlgnFashionMnistBudget8k",
                "16k": "DlgnFashionMnistBudget16k",
                "32k": "DlgnFashionMnistBudget32k",
                "64k": "DlgnFashionMnistBudget64k",
            },
            0.50,
        ),
    ]
    for prefix, dataset, budgets, swap_fraction in cells:
        for budget, architecture in budgets.items():
            for seed in range(3):
                for method in ("random", "coverage_v3"):
                    name = (
                        f"second_compression_{prefix}_{budget}_{method}_seed{seed}"
                    )
                    config = fixed_base(dataset, architecture, seed)
                    if method == "coverage_v3":
                        config.update({
                            "connections_init_method": (
                                "semantic_balanced_hybrid"
                            ),
                            "coverage_candidate_pool_size": 8,
                            "coverage_novelty_weight": 1.0,
                            "coverage_swap_fraction": swap_fraction,
                        })
                    entries.append((name, add_output(config, name)))
    return entries


def conv_s_full() -> list[tuple[str, dict]]:
    entries = []
    methods = {
        "random": "random",
        "legacy_v4": "semantic_channel_hybrid",
        "unified_u1": "semantic_degree_balanced",
    }
    for label, strategy in methods.items():
        name = f"second_full_conv_cifar10_s_{label}_seed0"
        config = conv_base(0)
        config.update({
            "num_iterations": 350_000,
            "conv_connections_init_method": strategy,
            "classifier_connections_init_method": "random",
        })
        if strategy != "random":
            config.update({
                "connections_init_method": strategy,
                "coverage_candidate_pool_size": 8,
                "coverage_novelty_weight": 1.0,
                "coverage_swap_fraction": (
                    0.25 if strategy == "semantic_channel_hybrid" else 0.0
                ),
            })
        entries.append((name, add_output(config, name)))
    return entries


def c100_strength() -> list[tuple[str, dict]]:
    entries = []
    base_arch = "DlgnCifar100Budget384kDepth3"
    # Baseline-only screen. The existing tau=10, lr=0.01, no-augmentation
    # seed-0 result is reused and therefore deliberately not emitted.
    for tau, learning_rate, augmentation in [
        (1.0, 0.01, "none"),
        (5.0, 0.01, "none"),
        (20.0, 0.01, "none"),
        (10.0, 0.02, "none"),
        (10.0, 0.01, "standard"),
        (10.0, 0.02, "standard"),
    ]:
        label = f"tau{tau:g}_lr{learning_rate:g}_aug{augmentation}"
        name = f"second_c100_baseline_screen_{label}_seed0"
        config = fixed_base("cifar-100", base_arch, 0)
        config.update({
            "num_iterations": 5_000,
            "eval_freq": 1_000,
            "group_sum_temperature": tau,
            "learning_rate": learning_rate,
            "augmentation": augmentation,
        })
        entries.append((name, add_output(config, name)))

    # Complete the three-seed 20K 3x128K pair without repeating seed zero.
    for seed in (1, 2):
        for method in ("random", "coverage_v3"):
            name = f"second_c100_3x128k_{method}_seed{seed}"
            config = fixed_base("cifar-100", base_arch, seed)
            config.update({"num_iterations": 20_000})
            if method == "coverage_v3":
                config.update({
                    "connections_init_method": "semantic_balanced_hybrid",
                    "coverage_candidate_pool_size": 8,
                    "coverage_novelty_weight": 1.0,
                    "coverage_swap_fraction": 0.25,
                })
            entries.append((name, add_output(config, name)))

    for architecture, label in [
        ("DlgnCifar100Budget384kClassScaled", "96_96_192k"),
        ("DlgnCifar100Budget384kClassHeavy", "64_64_256k"),
    ]:
        for method in ("random", "coverage_v3"):
            name = f"second_c100_class_scaled_{label}_{method}_seed0"
            config = fixed_base("cifar-100", architecture, 0)
            config.update({"num_iterations": 20_000})
            if method == "coverage_v3":
                config.update({
                    "connections_init_method": "semantic_balanced_hybrid",
                    "coverage_candidate_pool_size": 8,
                    "coverage_novelty_weight": 1.0,
                    "coverage_swap_fraction": 0.25,
                })
            entries.append((name, add_output(config, name)))
    return entries


def unified_u2_pilot() -> list[tuple[str, dict]]:
    entries = []
    dense_cells = [
        ("mnist_8k", "mnist", "DlgnMnistBudget8k", 20_000),
        (
            "fashion_16k",
            "fashion-mnist",
            "DlgnFashionMnistBudget16k",
            20_000,
        ),
        (
            "cifar10_s",
            "cifar-10",
            "DlgnCifar10Small",
            20_000,
        ),
        (
            "cifar100_3x128k",
            "cifar-100",
            "DlgnCifar100Budget384kDepth3",
            20_000,
        ),
    ]
    for label, dataset, architecture, iterations in dense_cells:
        for seed in range(3):
            name = f"second_u2_{label}_seed{seed}"
            config = fixed_base(dataset, architecture, seed)
            config.update({
                "num_iterations": iterations,
                "connections_init_method": "semantic_multiscale_balanced",
            })
            entries.append((name, add_output(config, name)))

    for seed in range(3):
        name = f"second_u2_conv_cifar10_s_seed{seed}"
        config = conv_base(seed)
        config.update({
            "connections_init_method": "semantic_multiscale_balanced",
            "conv_connections_init_method": "semantic_multiscale_balanced",
            "classifier_connections_init_method": (
                "semantic_multiscale_balanced"
            ),
        })
        entries.append((name, add_output(config, name)))
    return entries


def unified_u2_final() -> list[tuple[str, dict]]:
    """Full-effort runs for U2 coordinates promoted by the frozen pilots.

    U2 itself is unchanged.  Dense CIFAR-100 is intentionally absent because
    its three-seed pilot was neutral versus random and worse than frozen V3.
    The convolutional full run is seed zero, matching the existing 350K
    random/V4/U1 resource cohort; its three-seed evidence remains the 20K
    pilot.
    """
    entries = []
    dense_cells = [
        ("mnist_8k", "mnist", "DlgnMnistBudget8k"),
        (
            "fashion_16k",
            "fashion-mnist",
            "DlgnFashionMnistBudget16k",
        ),
        ("cifar10_s", "cifar-10", "DlgnCifar10Small"),
    ]
    for label, dataset, architecture in dense_cells:
        for seed in range(3):
            name = f"second_final_u2_{label}_seed{seed}"
            config = fixed_base(dataset, architecture, seed)
            config.update({
                "connections_init_method": "semantic_multiscale_balanced",
            })
            entries.append((name, add_output(config, name)))

    conv_name = "second_final_u2_conv_cifar10_s_seed0"
    conv = conv_base(0)
    conv.update({
        "num_iterations": 350_000,
        "connections_init_method": "semantic_multiscale_balanced",
        "conv_connections_init_method": "semantic_multiscale_balanced",
        "classifier_connections_init_method": "semantic_multiscale_balanced",
    })
    entries.append((conv_name, add_output(conv, conv_name)))
    return entries


def u2_post_balance_smoke() -> list[tuple[str, dict]]:
    """CUDA-only preflight for the corrected U2 matching invariant."""
    dense_name = "smoke_second_u2_mnist_8k_seed0_post_balance"
    dense = fixed_base("mnist", "DlgnMnistBudget8k", 0)
    dense.update({
        "num_iterations": 100,
        "eval_freq": 100,
        "connections_init_method": "semantic_multiscale_balanced",
    })

    conv_name = "smoke_second_u2_conv_cifar10_s_seed0_post_balance"
    conv = conv_base(0)
    conv.update({
        "num_iterations": 100,
        "eval_freq": 100,
        "connections_init_method": "semantic_multiscale_balanced",
        "conv_connections_init_method": "semantic_multiscale_balanced",
        "classifier_connections_init_method": "semantic_multiscale_balanced",
    })
    return [
        (dense_name, add_output(dense, dense_name)),
        (conv_name, add_output(conv, conv_name)),
    ]


PHASES = {
    "exact_comparators": exact_comparators,
    "compression": compression,
    "conv_s_full": conv_s_full,
    "c100_strength": c100_strength,
    "unified_u2_pilot": unified_u2_pilot,
    "unified_u2_final": unified_u2_final,
    "u2_post_balance_smoke": u2_post_balance_smoke,
}


def write_phase(phase: str) -> None:
    entries = PHASES[phase]()
    config_dir = CONFIG_ROOT / phase
    config_dir.mkdir(parents=True, exist_ok=True)
    expected = set()
    queue_entries = []
    for name, config in entries:
        path = config_dir / f"{name}.json"
        expected.add(path)
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        queue_entries.append({
            "name": name,
            "config": str(path.relative_to(REPO)),
            "output": config["output"],
        })
    stale = sorted(set(config_dir.glob("*.json")) - expected)
    if stale:
        raise RuntimeError(
            "refusing to remove stale configs: "
            + ", ".join(str(path) for path in stale)
        )
    queue = {
        "phase": f"second_round_{phase}",
        "cuda_required": True,
        "skip_completed": True,
        "entries": queue_entries,
    }
    queue_path = QUEUE_ROOT / f"second_round_{phase}.json"
    queue_path.write_text(json.dumps(queue, indent=2) + "\n")
    print(f"{phase}: wrote {len(entries)} configs to {queue_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phases", nargs="*", choices=sorted(PHASES))
    args = parser.parse_args()
    phases = args.phases or list(PHASES)
    for phase in phases:
        write_phase(phase)


if __name__ == "__main__":
    main()
