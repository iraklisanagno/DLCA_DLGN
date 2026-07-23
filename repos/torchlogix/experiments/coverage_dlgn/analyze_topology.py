#!/usr/bin/env python3
"""Generate CoverageDLGN topologies and report metrics without training."""

import argparse
import json
from pathlib import Path

from torchlogix.topology import (
    generate_dense_stack,
    image_input_semantics,
    strategy_choices,
    write_topology_report,
)


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--in-dim", type=int, default=784)
    parser.add_argument("--widths", type=int, nargs="+", default=[1000] * 5)
    parser.add_argument("--strategy", choices=strategy_choices(), default="coverage_hybrid")
    parser.add_argument("--topology-seed", type=int, default=0)
    parser.add_argument("--candidate-pool-size", type=int, default=64)
    parser.add_argument("--long-range-fraction", type=float, default=0.25)
    parser.add_argument("--swap-fraction", type=float, default=0.25)
    parser.add_argument("--novelty-weight", type=float, default=1.0)
    parser.add_argument("--output-groups", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--local-radius", type=int, default=4)
    parser.add_argument(
        "--hybrid-base", choices=["butterfly", "local_cyclic"], default="butterfly"
    )
    parser.add_argument(
        "--image-shape",
        type=int,
        nargs=3,
        metavar=("CHANNELS", "HEIGHT", "WIDTH"),
        default=None,
    )
    parser.add_argument("--threshold-bits", type=int, default=1)
    parser.add_argument(
        "--input-layout",
        choices=["pixel_interleaved", "channel_interleaved"],
        default="pixel_interleaved",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def parse_args(argv=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    pre_args, _ = pre_parser.parse_known_args(argv)
    parser = get_parser()
    if pre_args.config:
        with pre_args.config.open() as handle:
            defaults = json.load(handle)
        valid = {action.dest for action in parser._actions}
        unknown = sorted(set(defaults) - valid)
        if unknown:
            parser.error(f"Unknown configuration keys: {', '.join(unknown)}")
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.output is None:
        parser.error("--output is required either on the command line or in --config")
    args.output = Path(args.output)
    return args


def main(argv=None):
    args = parse_args(argv)
    semantics = (
        image_input_semantics(
            *args.image_shape,
            args.threshold_bits,
            layout=args.input_layout,
        )
        if args.image_shape is not None else None
    )
    if semantics is not None and semantics.n_inputs != args.in_dim:
        raise ValueError(
            f"--in-dim={args.in_dim} does not match semantic image inputs "
            f"({semantics.n_inputs})"
        )
    _, rows = generate_dense_stack(
        args.in_dim,
        args.widths,
        strategy=args.strategy,
        topology_seed=args.topology_seed,
        candidate_pool_size=args.candidate_pool_size,
        long_range_fraction=args.long_range_fraction,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        local_radius=args.local_radius,
        hybrid_base=args.hybrid_base,
        input_semantics=semantics,
        swap_fraction=args.swap_fraction,
        output_groups=args.output_groups,
        novelty_weight=args.novelty_weight,
    )
    write_topology_report(rows, args.output, metadata=vars(args))
    for row in rows:
        print(
            f"depth={row['depth']} coverage={row['input_coverage']:.4f} "
            f"mean_ancestry={row['mean_gate_ancestry']:.2f} "
            f"overlap={row['overlap_mean']:.2f} fanout_cv={row['fanout_cv']:.3f} "
            f"distinct_pairs={row['distinct_predecessor_pairs']}"
        )


if __name__ == "__main__":
    main()
