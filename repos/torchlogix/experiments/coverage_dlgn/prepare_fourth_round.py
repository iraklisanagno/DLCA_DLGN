"""Declare the bounded, validation-only fourth round; never overwrite artifacts."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
U2 = "semantic_multiscale_balanced"
METHODS = {"random": "random", "balanced_random": "semantic_random_balanced",
           "nominal": "semantic_multiscale_nominal", "u2": U2}


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path: Path, payload):
    """Allow idempotent preparation, never replace different existing content."""
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"refusing to change existing artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(text)


def base(coordinate, seed):
    name = (f"pilot_cifar10_medium_random_v3_seed{seed}" if coordinate == "dense_m"
            else f"pilot_conv_cifar10_paper_small_random_controlled_seed{seed}")
    config = read(ROOT / "results" / name / "training_config.json")
    config.pop("config", None)
    config.pop("output", None)
    config["classifier_reference_u2"] = False
    return config, name


def matrix():
    rows = []

    def add(phase, coordinate, family, seed, config, reuse=None):
        name = f"fourth_{phase}_{coordinate}_{family}_seed{seed}"
        output = ROOT / "results" / (reuse or name)
        config = dict(config, output=str(output.relative_to(REPO)))
        if reuse:
            # Legacy configs differ on this unused learned-routing option.
            # Retain their actual value, rather than misstate archived metadata.
            archived = read(output / "training_config.json")
            assert config["connections"] == archived["connections"] == "fixed"
            config["connections_gumbel"] = archived["connections_gumbel"]
        rows.append(dict(name=name, phase=phase, coordinate=coordinate, family=family,
                         seed=seed, reuse=bool(reuse), output=str(output.relative_to(REPO)),
                         config=config, scope="V", provenance="ADAPTED" if phase.startswith("warp_")
                         else "OUR" if family != "random" else "REPRODUCED"))

    for coordinate in ["dense_m", "conv_s"]:
        for seed in range(3):
            original, random_name = base(coordinate, seed)
            for family, strategy in METHODS.items():
                cfg = dict(original, connections_init_method=strategy)
                if coordinate == "conv_s":
                    cfg.update(conv_connections_init_method=strategy, classifier_connections_init_method=strategy)
                reuse = random_name if family == "random" else None
                if coordinate == "conv_s" and family == "u2":
                    reuse = f"second_u2_conv_cifar10_s_seed{seed}"
                add("mechanism", coordinate, family, seed, cfg, reuse)
    for seed in range(3):
        original, random_name = base("conv_s", seed)
        for family, body, head in [("random", "random", "random"), ("body", U2, "random"),
                                    ("head", "random", U2), ("both", U2, U2)]:
            cfg = dict(original, connections_init_method=body, conv_connections_init_method=body,
                       classifier_connections_init_method=head, classifier_reference_u2=True)
            reuse = random_name if family == "random" else f"second_u2_conv_cifar10_s_seed{seed}" if family == "both" else None
            add("factorial", "conv_s", family, seed, cfg, reuse)
    # A faithful implementation compatibility test, adapted to our frozen dense
    # protocol: only parametrization changes relative to its raw counterpart.
    # It is NOT a reproduction of every WARP-paper architecture/training choice.
    for phase, coordinate, seeds, steps in [
        ("warp_pilot", "dense_m", [0], 20_000),
        ("warp_confirm", "dense_m", [0, 1, 2], 108_000),
        ("warp_conv", "conv_s", [0, 1, 2], 20_000),
    ]:
        for seed in seeds:
            original, _ = base(coordinate, seed)
            for family, strategy in [("random", "random"), ("u2", U2)]:
                cfg = dict(original, parametrization="warp", connections_init_method=strategy,
                           num_iterations=steps)
                if coordinate == "conv_s":
                    cfg.update(conv_connections_init_method=strategy, classifier_connections_init_method=strategy)
                add(phase, coordinate, family, seed, cfg)
    return rows


def prepare():
    entries = []
    for row in matrix():
        row = dict(row)
        cfg = row.pop("config")
        path = ROOT / "configs/fourth_round" / f"{row['name']}.json"
        write_new(path, cfg)
        row["config"] = str(path.relative_to(REPO))
        entries.append(row)
    references = []
    for row in matrix():
        if row["phase"] != "warp_confirm":
            continue
        seed = row["seed"]
        name = (f"paper_cifar10_medium_random_seed{seed}" if row["family"] == "random"
                else f"third_u2_cifar10_m_seed{seed}")
        output = ROOT / "results" / name
        cfg_path = output / "training_config.json"
        cfg = read(cfg_path)
        for key, value in row["config"].items():
            if key not in {"config", "output", "parametrization", "classifier_reference_u2", "connections_gumbel"}:
                if cfg.get(key) != value:
                    raise RuntimeError(f"unmatched raw WARP reference {name}: {key}")
        if cfg["parametrization"] != "raw" or cfg["connections"] != "fixed":
            raise RuntimeError("raw reference is not a fixed raw model")
        references.append(dict(row, name=name, phase="raw_reference", reuse=True,
                               config=str(cfg_path.relative_to(REPO)), output=str(output.relative_to(REPO)),
                               provenance="OUR" if row["family"] == "u2" else "REPRODUCED"))
    payload = {"protocol": "FOURTH_ROUND_PROTOCOL.md", "entries": entries,
               "raw_warp_references": references,
               "hardware": False, "extra_full_conv_s_seeds": False,
               "warp_promotion": {"pilot_random_min_hard_validation_pct": 40.,
                                  "pilot_u2_minus_random_min_pp": 0.,
                                  "strictly_positive_gain_required": True},
               "heldout_cifar10_access": False}
    write_new(ROOT / "protocols/fourth_round.json", payload)
    return payload


if __name__ == "__main__":
    p = prepare()
    print(f"Declared {len(p['entries'])} cells; {sum(not r['reuse'] for r in p['entries'])} new conditional runs")
