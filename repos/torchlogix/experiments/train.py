#!/usr/bin/env python3
"""Training script for TorchLogix models."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
from typing import Optional
from dataclasses import dataclass

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import torch
from tqdm import tqdm
import torchlogix
import torchlogix.models
from torchlogix.topology import (
    analyze_conv_channel_topology,
    analyze_model_topology,
    model_topology_metadata,
    strategy_choices,
    write_topology_report,
)

try:
    from .utils import (
        CreateFolder, save_metrics_csv, save_config, save_thresholds_csv,
        evaluate_model, get_model, load_dataset, load_n
    )
except ImportError:  # Direct execution: python experiments/train.py
    from utils import (
        CreateFolder, save_metrics_csv, save_config, save_thresholds_csv,
        evaluate_model, get_model, load_dataset, load_n
    )

def get_parser():
    parser = argparse.ArgumentParser(description="Train TorchLogix models")
    parser.add_argument(
        "--config", type=Path, default=None,
        help="JSON file providing parser defaults; explicit CLI arguments take precedence"
    )
    # Dataset and architecture
    parser.add_argument(
        "--dataset", type=str, choices=["mnist", "fashion-mnist", "cifar-10"],
        default="mnist", help="Dataset to train on"
    )
    parser.add_argument(
        "--architecture", "-a", choices=torchlogix.models.__dict__.keys(),
        default="DlgnMnistSmall", help="Model architecture. Must match dataset"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"],
        help="Device to use (cuda is faster)"
    )

    # Training parameters
    parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--data-split-seed", type=int, default=None,
        help=(
            "Independent train/validation split seed. If omitted, preserve the "
            "legacy behavior in which --seed and the global PyTorch RNG control the split."
        ),
    )
    parser.add_argument("--batch-size", "-bs", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--num-iterations", "-ni", type=int, default=100_000, help="Number of training iterations"
    )
    parser.add_argument(
        "--eval-freq", "-ef", type=int, default=2_000, 
        help="Evaluation frequency. Evaluation is deactivated if set to 0."
    )
    parser.add_argument(
        "--valid-set-size", "-vss", type=float, default=0.1,
        help="Fraction of train set for validation"
    )
    parser.add_argument(
        "--calibration-set-size", type=float, default=0.0,
        help=(
            "Fraction of the official training split reserved for post-training "
            "calibration and excluded from model training and validation"
        ),
    )
    parser.add_argument(
        "--augmentation", choices=["none", "standard"], default="none",
        help="Training-only data augmentation; standard means crop/flip for CIFAR"
    )

    # Learning rate parameters
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--lr-schedule", type=str, choices=[None, "CosineAnnealingWarmRestarts", "ReduceLROnPlateau"],
        default=None, help="Learning rate scheduling strategy"
    )
    parser.add_argument(
        "--lr-reduction-factor", "-lrf", type=float, default=0.2,
        help="If lr-schedule is ReduceLROnPlateau, factor by which LR will be reduced. "
    )
    parser.add_argument(
        "--lr-patience", "-lrp", type=int, default=10, 
        help="If lr-schedule is ReduceLROnPlateau, patience for LR reduction." \
             "If CosineAnnealingWarmRestarts, length of each cycle." \
             "Counted in number of evaluations."
    )
    parser.add_argument("--half-precision", action="store_true", 
                        help="Use half-precision (bfloat16) training to reduce memory usage and speed up training")
    parser.add_argument("--compile-model", action="store_true", 
                        help="Use toch.compile() to compile the model for faster training")

    parser.add_argument(
        "--output", "-o", action=CreateFolder, type=Path, default="results/training/",
        help="Output directory for results"
    )
    parser.add_argument(
        "--verbose", type=int, default=0, choices=[0, 1],
        help="Verbosity during training, allowed only for lut_rank=2. 0 = silent, 1 = verbose"
    )
    parser.add_argument(
        "--weight-decay", "-wd", type=float, default=None, help="Weight decay for optimizer"
    )

    # Connection parameters
    parser.add_argument(
        "--connections", type=str, choices=["fixed", "learnable"],
        default="fixed", help="Connection strategy"
    )
    parser.add_argument(
        "--connections-init-method", type=str, choices=strategy_choices(),
        default="random", help="Connection initialization strategy"
    )
    parser.add_argument(
        "--topology-seed", type=int, default=None,
        help="Independent fixed-topology seed. Omit to preserve legacy random initialization."
    )
    parser.add_argument(
        "--coverage-candidate-pool-size", type=int, default=64,
        help="Number of candidate predecessor pairs scored per greedy edge"
    )
    parser.add_argument(
        "--coverage-long-range-fraction", type=float, default=0.25,
        help="Fraction of hybrid gates assigned greedy long-range pairs"
    )
    parser.add_argument(
        "--coverage-swap-fraction", type=float, default=0.25,
        help=(
            "Fraction of gates eligible for degree-preserving semantic-hybrid "
            "two-edge swaps"
        ),
    )
    parser.add_argument(
        "--coverage-novelty-weight", type=float, default=1.0,
        help="Cross-gate semantic-ancestry novelty weight for balanced swaps",
    )
    parser.add_argument("--coverage-alpha", type=float, default=1.0)
    parser.add_argument("--coverage-beta", type=float, default=1.0)
    parser.add_argument("--coverage-gamma", type=float, default=0.25)
    parser.add_argument("--coverage-delta", type=float, default=0.0)
    parser.add_argument("--coverage-local-radius", type=int, default=4)
    parser.add_argument(
        "--coverage-hybrid-base", choices=["butterfly", "local_cyclic"],
        default="butterfly"
    )
    parser.add_argument(
        "--connections-temperature", type=float, default=0.001,
        help="Temperature for softmax in learnable connections"
    )
    parser.add_argument(
        "--connections-gumbel", action="store_false", 
        help="Flag for using Gumbel sampling for softmax. "
    )

    # Parametrization parameters
    parser.add_argument(
        "--lut-rank", type=int, default=2, choices=[2, 4, 6],
        help="Number of inputs to each LUT node"
    )
    parser.add_argument(
        "--parametrization", type=str, default="raw", choices=["raw", "warp", "light"],
        help="Parametrization to use"
    )
    parser.add_argument(
        "--parametrization-temperature", type=float, default=1.0,
        help="Temperature for sigmoid/softmax in parametrization"
    )
    parser.add_argument(
        "--forward-sampling", type=str, default="soft", choices=["soft", "hard", "gumbel_soft", "gumbel_hard"],
        help="Sampling method in forward pass during training"
    )
    parser.add_argument(
        "--weight-init", type=str, default="residual", choices=["residual", "random", "residual-catalog"],
        help="Initialization method for model weights"
    )
    parser.add_argument(
        "--residual-probability", type=float, default=0.951,
        help="Parameter for residual weight initialization. " \
        "Corresponds to probability of a LUT entry corresponding to identity LUT entry."
    )

    # Binarization parameters
    parser.add_argument(
        "--binarization-num-batches", type=int, default=100,
        help="Number of batches for initializing thresholds in binarization"
    )
    parser.add_argument(
        "--binarization", type=str, default="fixed", choices=["dummy", "fixed", "soft", "learnable"],
        help="Binarization method for input data"
    )
    parser.add_argument(
        "--binarization-init", type=str, default="uniform", choices=["uniform", "distributive"],
        help="Method to find initial thresholds for binarization"
    )
    parser.add_argument(
        "--binarization-per", type=str, default="global", choices=["global", "feature", "channel"],
        help="Binarization thresholds global, per channel, or per feature"
    )
    parser.add_argument(
        "--binarization-temperature", type=float, default=0.001,
        help="Temperature for sampling in learnable binarization"
    )
    parser.add_argument(
        "--binarization-temperature-softplus", type=float, default=0.01,
        help="Temperature for softplus in learnable binarization"
    )
    parser.add_argument(
        "--binarization-learning-rate", type=float, default=None,
        help="Learning rate for binarization (as fraction of main learning rate). If None, uses main learning rate."
    )
    parser.add_argument(
        "--binarization-forward-sampling", type=str, default="soft", choices=["soft", "hard", "gumbel_soft", "gumbel_hard"],
        help="Sampling method in forward pass during training for learnable binarization"
    )

    return parser


@dataclass
class CallbackContext:
    """Context passed to training callbacks."""
    step: int
    metrics: dict  # Required: val_loss, train_loss, etc.
    model: Optional[torch.nn.Module] = None  # Optional for advanced use
    args: Optional[argparse.Namespace] = None


class LearningRateSchedulerCallback:
    """Wrapper for learning rate schedulers to be used as callbacks."""
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def __call__(self, ctx: CallbackContext):
        if self.scheduler is None:
            pass
        elif isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(ctx.metrics["val_loss_discrete"])
        elif isinstance(self.scheduler, CosineAnnealingWarmRestarts):
            self.scheduler.step()

    @classmethod
    def from_args(cls, optimizer, args):
        if args.lr_schedule is None:
            scheduler = None
        elif args.lr_schedule == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=args.lr_reduction_factor, patience=args.lr_patience#, verbose=False
            )
        elif args.lr_schedule == "CosineAnnealingWarmRestarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=args.lr_patience, T_mult=1, eta_min=0.0, last_epoch=-1
            )
        else:
            raise ValueError(f"Unknown learning rate schedule: {args.lr_schedule}")
        return cls(scheduler)


def save_best_model(ctx: CallbackContext, output_dir: Path):
    """Callback to save the best model based on validation accuracy."""
    val_acc = ctx.metrics.get("val_acc_discrete", 0.0)
    if not hasattr(save_best_model, "best_val_acc"):
        save_best_model.best_val_acc = 0.0

    if val_acc > save_best_model.best_val_acc:
        save_best_model.best_val_acc = val_acc
        model_path = f"{output_dir}/best_model.pt"
        torch.save(ctx.model.state_dict(), model_path)
        torch.save(
            checkpoint_payload(ctx.model, ctx.args, ctx.step, ctx.metrics),
            f"{output_dir}/best_checkpoint.pt",
        )
        print(f"New best model saved with val_accuracy: {val_acc:.4f} at step {ctx.step}")


def checkpoint_payload(model, args, step, metrics):
    """Create a self-describing checkpoint while retaining legacy state dict files."""
    def plain(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            return value.detach().cpu().item()
        if isinstance(value, dict):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    return {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "metadata": {
            "step": plain(step),
            "metrics": plain(dict(metrics)),
            "configuration": plain(vars(args)) if args is not None else {},
            "topology": plain(model_topology_metadata(model)),
            "python": platform.python_version(),
            # ``torch.__version__`` is a TorchVersion (a ``str`` subclass), which
            # PyTorch's default weights-only loader intentionally rejects.
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
        },
    }


def source_manifest_files(root: Path = Path(".")):
    """Return code/config files, excluding generated experiment artifacts."""
    files = sorted((root / "src" / "torchlogix").rglob("*.py"))
    files += sorted((root / "experiments").rglob("*.py"))
    files += [
        path
        for path in sorted((root / "experiments").rglob("*.json"))
        if not {"results", "summary"}.intersection(path.relative_to(root).parts)
    ]
    return files


def dense_architecture_summary(model, architecture_name: str) -> dict:
    """Describe dense logic depth and gate count in logs and run metadata."""
    layers = [
        module
        for module in model.modules()
        if isinstance(module, torchlogix.layers.LogicDense)
    ]
    widths = [int(layer.out_dim) for layer in layers]
    ranks = sorted({int(layer.lut_rank) for layer in layers})
    return {
        "architecture": architecture_name,
        "logic_layer_count": len(layers),
        "logic_layer_widths": widths,
        "total_trained_logic_gates": sum(widths),
        "lut_ranks": ranks,
    }


def source_tree_sha256(root: Path = Path(".")):
    """Hash stable source-relative paths and contents."""
    source_hasher = hashlib.sha256()
    for path in source_manifest_files(root):
        source_hasher.update(path.relative_to(root).as_posix().encode())
        source_hasher.update(path.read_bytes())
    return source_hasher.hexdigest()


def training_manifest_files(root: Path = Path(".")):
    """Return only implementation files imported by the training path."""
    files = sorted((root / "src" / "torchlogix").rglob("*.py"))
    files += [
        root / "experiments" / name
        for name in ("train.py", "utils.py")
        if (root / "experiments" / name).exists()
    ]
    return files


def training_implementation_sha256(root: Path = Path(".")):
    """Hash training code without unrelated reports, queues, or configs."""
    hasher = hashlib.sha256()
    for path in training_manifest_files(root):
        hasher.update(path.relative_to(root).as_posix().encode())
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def save_environment_fingerprint(output_dir: Path):
    """Record enough source and environment state to reproduce a run."""
    def git(*arguments):
        result = subprocess.run(
            ["git", *arguments], capture_output=True, text=True, check=False
        )
        return result.stdout.strip()

    payload = {
        "source_revision": git("rev-parse", "HEAD"),
        "source_status": git("status", "--short"),
        "source_tree_sha256": source_tree_sha256(),
        "source_manifest_scope": (
            "src/torchlogix/**/*.py, experiments/**/*.py, and "
            "experiments/**/*.json excluding results/ and summary/"
        ),
        "training_implementation_sha256": training_implementation_sha256(),
        "training_manifest_scope": (
            "src/torchlogix/**/*.py, experiments/train.py, and "
            "experiments/utils.py; the resolved run configuration is stored "
            "separately in training_config.json"
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_names": (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else []
        ),
        "packages": {
            distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
            if distribution.metadata["Name"]
        },
    }
    with (Path(output_dir) / "environment.json").open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def run_training(args, callbacks=None):
    """Run the training loop."""
    if callbacks is None:
        callbacks = []
    save_best_model.best_val_acc = 0.0
    # Setup experiment
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    torch.set_num_threads(1)

    # Load data (omit test set during training)
    train_loader, validation_loader, calibration_loader, _ = load_dataset(
        args, include_calibration=True
    )

    # Initial thresholds
    data_set = torch.cat(tuple([batch[0] for batch in load_n(train_loader, args.binarization_num_batches)]))
    model_cls = torchlogix.models.__dict__[args.architecture]
    thresholds = torchlogix.layers.Binarization.get_initial_thresholds(
        data_set,
        num_bits=model_cls.n_input_bits,
        one_per=args.binarization_per,
        method=args.binarization_init
    )

    print("Initial thresholds:", thresholds)

    # Get model, loss, and optimizer
    model= get_model(thresholds, args)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params}")
    architecture_summary = dense_architecture_summary(model, args.architecture)
    print(
        "Architecture summary: "
        f"{architecture_summary['architecture']}; "
        f"{architecture_summary['logic_layer_count']} logic layers; "
        f"widths={architecture_summary['logic_layer_widths']}; "
        f"total trained gates="
        f"{architecture_summary['total_trained_logic_gates']}; "
        f"LUT ranks={architecture_summary['lut_ranks']}"
    )

    topology_rows = analyze_model_topology(model)
    if args.output is not None and topology_rows:
        write_topology_report(
            topology_rows,
            args.output,
            metadata={
                "architecture": args.architecture,
                "strategy": args.connections_init_method,
                "topology_seed": args.topology_seed,
            },
        )
    conv_topology_rows = analyze_conv_channel_topology(model)
    if args.output is not None and conv_topology_rows:
        write_topology_report(
            conv_topology_rows,
            args.output,
            stem="conv_topology",
            metadata={
                "architecture": args.architecture,
                "strategy": args.connections_init_method,
                "topology_seed": args.topology_seed,
                "spatial_indexing": "unchanged",
            },
        )

    model.to(args.device)
    print(model)

    if args.compile_model:
        print("JIT compilation has been chosen.")
        print("The first iteration will be slower due to compilation overhead.")
        torch._dynamo.config.cache_size_limit = 64
        # Most aggressive optimization. May lead to long compilation times and OOM errors for large models.
        # Adjust settings if you encounter issues. E.g. dynamic=True can help
        model.compile(fullgraph=True, mode="max-autotune")

    # Loss function for classification tasks like MNIST and CIFAR-10
    loss_fn = torch.nn.CrossEntropyLoss()
    # Create evaluation functions
    eval_functions = {
        "loss": loss_fn,
        "acc": lambda preds, y: (preds.argmax(-1) == y).to(torch.float32).mean(),
    }

    # Set up optimizer with optional separate learning rate for binarization parameters
    params_list = []
    binarization_params = []
    if args.binarization_learning_rate and isinstance(model[0], torchlogix.layers.LearnableBinarization):
        binarization_params += list(model[0].parameters())
        params_list += [{'params': binarization_params, 'lr': args.binarization_learning_rate * args.learning_rate}]
    else:
        if args.binarization_learning_rate:
            print("Warning: binarization_learning_rate specified but the model does not use LearnableBinarization. Ignoring this parameter.")
    other_params = [p for p in model.parameters() if p not in set(binarization_params)]
    params_list += [{'params': other_params, 'lr': args.learning_rate}]

    if args.weight_decay is not None:
        # weight decay should not be applied to learnable binarization parameters
        # Would be nicer to implement this in the binarization layer itself (not sure if possible)
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():

            if not param.requires_grad:
                continue

            if "raw_diffs" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": args.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ]
        )
        optimizer = torch.optim.AdamW(params_list, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(params_list)

    # Training tracking
    metrics = defaultdict(dict)
    best_val_acc = 0.0
    started = time.perf_counter()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    learning_rate_scheduler = LearningRateSchedulerCallback.from_args(optimizer, args)
    callbacks.append(learning_rate_scheduler)

    print(f"Starting training for {args.num_iterations} iterations...")
    print(f"Model: {args.architecture}, Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}, Learning rate: {args.learning_rate}")
    
    if args.output is not None:
        save_config(vars(args), args.output, "training_config.json")
        save_environment_fingerprint(args.output)
        with (Path(args.output) / "data_split.json").open("w") as handle:
            json.dump(
                train_loader.split_manifest,
                handle,
                indent=2,
                sort_keys=True,
            )

    pbar = tqdm(
        enumerate(load_n(train_loader, args.num_iterations)),
        desc="Training",
        total=args.num_iterations,
        mininterval=1,
    )
    running_train_loss, n = 0.0, 0
    for i, (x, y) in pbar:
        x = x.to(args.device)
        y = y.to(args.device)

        dtype = torch.bfloat16 if args.half_precision else torch.float32
        with torch.amp.autocast("cuda", dtype=dtype):
            model.train()
            x = model(x)
            loss = loss_fn(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        n += y.size(0)
        running_train_loss += loss

        if i % 100 == 0:
            pbar.set_postfix(loss=f"{loss:.4f}")

        # Evaluation
        if (args.eval_freq > 0 and ((i + 1) % args.eval_freq == 0)):
            if args.verbose == 1:
                print(f"\nEvaluation at iteration {i + 1}")          

            # Evaluate on validation set
            discrete_metrics = evaluate_model(
                model, validation_loader, eval_functions, mode="eval", device=args.device
            )
            relaxed_metrics = evaluate_model(
                model, validation_loader, eval_functions, mode="train", device=args.device
            )

            metrics = \
                {f"val_{k}_discrete": v for k, v in discrete_metrics.items()} | \
                {f"val_{k}_relaxed": v for k, v in relaxed_metrics.items()} | \
                {"train_loss": running_train_loss.cpu().detach().item() / n * len(validation_loader)}
        
            print(f"Iteration {i + 1:6d} | " +
                  " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

            running_train_loss, n = 0.0, 0

            ctx = CallbackContext(
                step=i + 1,
                metrics=metrics,
                model=model,
                args=args,
            )

            best_val_acc = max(best_val_acc, float(metrics["val_acc_discrete"]))

            for cb in callbacks:
                cb(ctx)

    # Save final model
    if args.output is not None:
        torch.save(model.state_dict(), f"{args.output}/final_model.pt")
        torch.save(
            checkpoint_payload(model, args, args.num_iterations, metrics),
            f"{args.output}/final_checkpoint.pt",
        )
        wall_seconds = time.perf_counter() - started
        summary = {
            "wall_seconds": wall_seconds,
            "architecture": architecture_summary,
            "best_validation_hard_accuracy": best_val_acc,
            "final_metrics": metrics,
            "dataset_partition_sizes": {
                name: partition["size"]
                for name, partition in
                train_loader.split_manifest["partitions"].items()
            },
            "dataset_partition_hashes": {
                name: partition["indices_sha256"]
                for name, partition in
                train_loader.split_manifest["partitions"].items()
            },
            "peak_gpu_memory_bytes": (
                torch.cuda.max_memory_allocated() if args.device == "cuda" else 0
            ),
            "topology": model_topology_metadata(model),
        }
        with open(f"{args.output}/run_summary.json", "w") as handle:
            json.dump(summary, handle, indent=2, default=str)

    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Results saved to: {args.output}")

    return metrics


def parse_args(argv=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)
    parser = get_parser()
    if pre_args.config is not None:
        with pre_args.config.open() as handle:
            defaults = json.load(handle)
        valid_destinations = {action.dest for action in parser._actions}
        unknown = sorted(set(defaults) - valid_destinations)
        if unknown:
            parser.error(f"Unknown configuration keys: {', '.join(unknown)}")
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.output is not None:
        args.output = Path(args.output)
        args.output.mkdir(parents=True, exist_ok=True)
    return args


def main(argv=None):
    args = parse_args(argv)

    # Validation
    if args.eval_freq > 0:
        assert args.num_iterations % args.eval_freq == 0, (
            f"Number of iterations ({args.num_iterations}) must be divisible by "
            f"evaluation frequency ({args.eval_freq})"
        )

    call_backs = [
        lambda ctx: save_best_model(ctx, args.output),
        lambda ctx: save_metrics_csv(ctx.step, ctx.metrics, args.output),
        lambda ctx: save_thresholds_csv(ctx.step, thresholds=ctx.model[0].get_thresholds().detach(), 
                                        output_path=args.output) if hasattr(ctx.model[0], "get_thresholds") else None
    ]

    # Pretty print args
    print("Training configuration:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    run_training(args, call_backs)


if __name__ == "__main__":
    main()
