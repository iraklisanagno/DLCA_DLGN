import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict
import math

import numpy as np
import torch
import torchvision
import torchlogix
import torchlogix.models


def split_permutation(length: int, seed=None):
    """Return split indices without advancing the global RNG when seeded."""
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    return torch.randperm(length, generator=generator).tolist()


def load_dataset(args):
    """Load a public dataset."""
    # check env varaible for dataset path
    data_path = os.getenv("DATASET_PATH", ".")
    transform = torchvision.transforms.ToTensor()
    train_transform = transform
    if getattr(args, "augmentation", "none") == "standard":
        if args.dataset != "cifar-10":
            raise ValueError("standard augmentation is currently defined only for CIFAR-10")
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transform,
        ])
    if args.dataset == "mnist":     
        train_set = torchvision.datasets.MNIST(
            f"{data_path}/data-mnist", train=True, download=True, transform=train_transform
        )
        validation_source = torchvision.datasets.MNIST(
            f"{data_path}/data-mnist", train=True, transform=transform
        )
        test_set = torchvision.datasets.MNIST(
            f"{data_path}/data-mnist", train=False, transform=transform
        )
    elif args.dataset == "fashion-mnist":
        train_set = torchvision.datasets.FashionMNIST(
            f"{data_path}/data-fashion-mnist", train=True, download=True, transform=train_transform
        )
        validation_source = torchvision.datasets.FashionMNIST(
            f"{data_path}/data-fashion-mnist", train=True, transform=transform
        )
        test_set = torchvision.datasets.FashionMNIST(
            f"{data_path}/data-fashion-mnist", train=False, transform=transform
        )
    elif args.dataset == "cifar-10":
        train_set = torchvision.datasets.CIFAR10(
            f"{data_path}/data-cifar", train=True, download=True, transform=train_transform
        )
        validation_source = torchvision.datasets.CIFAR10(
            f"{data_path}/data-cifar", train=True, transform=transform
        )
        test_set = torchvision.datasets.CIFAR10(
            f"{data_path}/data-cifar", train=False, transform=transform
        )
    
    if args.valid_set_size > 0:
        train_set_size = math.ceil((1 - args.valid_set_size) * len(train_set))
        valid_set_size = len(train_set) - train_set_size
        split_seed = getattr(args, "data_split_seed", None)
        permutation = split_permutation(len(train_set), split_seed)
        train_set = torch.utils.data.Subset(train_set, permutation[:train_set_size])
        validation_set = torch.utils.data.Subset(
            validation_source, permutation[train_set_size:]
        )
    else:
        print(f"Training on entire training set. Using test set as validation set.")
        validation_set = test_set


    train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=0,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, validation_loader, test_loader


def load_n(loader, n):
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i == n:
                break


def input_threshold_count(model_cls):
    """Return the number of Boolean thresholds required by a model class.

    Legacy TorchLogix models use ``n_input_bits`` for this quantity.  The
    convolutional DLGN paper describes S/M as 2-bit inputs encoded by three
    thermometer thresholds, so paper-faithful classes expose the unambiguous
    ``n_input_thresholds`` attribute instead.
    """
    threshold_count = getattr(model_cls, "n_input_thresholds", None)
    if threshold_count is None:
        threshold_count = getattr(model_cls, "n_input_bits", None)
    if threshold_count is None:
        raise AttributeError(
            f"{model_cls.__name__} does not declare an input threshold count"
        )
    return int(threshold_count)


def get_model(thresholds, args):
    """
    Select model from the architecture.
    It can be a difflogic model or a baseline model.
    """
    llkw = {
        "connections": args.connections,
        "connections_kwargs": {
            "init_method": args.connections_init_method,
            "conv_init_method": getattr(
                args, "conv_connections_init_method", None
            ),
            "classifier_init_method": getattr(
                args, "classifier_connections_init_method", None
            ),
            "temperature": args.connections_temperature,
            "gumbel": args.connections_gumbel,
            "num_candidates": getattr(args, "connections_num_candidates", -1),
            "forward_mode": getattr(
                args, "connections_forward_mode", "hard_st"
            ),
            "weights_init": getattr(
                args, "connections_weights_init", "uniform"
            ),
            "topology_seed": getattr(args, "topology_seed", None),
            "candidate_pool_size": getattr(args, "coverage_candidate_pool_size", 64),
            "long_range_fraction": getattr(args, "coverage_long_range_fraction", 0.25),
            "coverage_alpha": getattr(args, "coverage_alpha", 1.0),
            "coverage_beta": getattr(args, "coverage_beta", 1.0),
            "coverage_gamma": getattr(args, "coverage_gamma", 0.25),
            "coverage_delta": getattr(args, "coverage_delta", 0.0),
            "local_radius": getattr(args, "coverage_local_radius", 4),
            "hybrid_base": getattr(args, "coverage_hybrid_base", "butterfly"),
            "swap_fraction": getattr(args, "coverage_swap_fraction", 0.25),
            "novelty_weight": getattr(args, "coverage_novelty_weight", 1.0),
            "reuse_change_fraction": getattr(
                args, "coverage_reuse_change_fraction", 0.25
            ),
            "reuse_weight": getattr(args, "coverage_reuse_weight", 1.0),
            },
        "parametrization": args.parametrization,
        "parametrization_kwargs": {
            "temperature": args.parametrization_temperature,
            "forward_sampling": args.forward_sampling,
            "weight_init": args.weight_init,
            "residual_probability": args.residual_probability,
            },
        "device": args.device,
        "lut_rank": args.lut_rank,
        "thresholds": thresholds,
        "binarization": args.binarization,
        "binarization_kwargs": {
            "one_per": args.binarization_per,
            "temperature_sampling": args.binarization_temperature,
            "temperature_softplus": args.binarization_temperature_softplus,
            "forward_sampling": args.binarization_forward_sampling
            }
    }
    model_cls = torchlogix.models.__dict__[args.architecture]
    model = model_cls(**llkw)
    group_sum_temperature = getattr(args, "group_sum_temperature", None)
    if group_sum_temperature is not None:
        group_sums = [
            module
            for module in model.modules()
            if isinstance(module, torchlogix.layers.GroupSum)
        ]
        if len(group_sums) != 1:
            raise ValueError(
                "group_sum_temperature requires exactly one GroupSum module"
            )
        group_sums[0].tau = float(group_sum_temperature)
    return model

class CreateFolder(argparse.Action):
    """
    Custom action: create a new folder if not exist. If the folder
    already exists, do nothing.

    The action will strip off trailing slashes from the folder's name.
    """

    def create_folder(self, folder_name):
        """
        Create a new directory if not exist, including parent directories.
        The action might throw OSError, along with other kinds of exception
        """
        # Create all parent directories if they don't exist
        os.makedirs(folder_name, exist_ok=True)

        folder_name = os.path.normpath(folder_name)
        return folder_name

    def __call__(self, parser, namespace, values, option_string=None):
        if type(values) == list:
            folders = list(map(self.create_folder, values))
        else:
            folders = self.create_folder(values)
        setattr(namespace, self.dest, folders)

# Shared experiment utilities
def save_metrics_csv(step: int, metrics: Dict[str, Any], output_path: Path, filename: str = "metrics.csv"):
    """Append single step metrics to CSV file."""
    filepath = f"{output_path}/{filename}"
    
    # Determine if file exists to write headers
    file_exists = Path(filepath).exists()

    # Convert tensor/numpy values to Python primitives
    row = {'step': step}
    for key, value in metrics.items():
        if hasattr(value, 'item'):  # numpy/torch scalar
            row[key] = value.item()
        else:
            row[key] = value
    
    fieldnames = ['step'] + sorted(metrics.keys())
    
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=fieldnames, lineterminator="\n"
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_thresholds_csv(
    step: int, 
    thresholds: torch.Tensor, 
    output_path: Path, 
    filename: str = "thresholds.csv"
):
    """Append threshold values to CSV file.
    
    Args:
        step: Training step number
        thresholds: Threshold tensor with last dimension = num_bits
                   Shapes can be (num_bits,), (num_features, num_bits), 
                   (num_channels, num_bits), or higher dimensional
        output_path: Directory to save CSV
        filename: Name of CSV file
    """
    filepath = f"{output_path}/{filename}"
    file_exists = Path(filepath).exists()

    # Flatten thresholds into a single row
    row = {'step': step}
    
    # Convert to numpy for easier indexing
    thresh_np = thresholds.detach().cpu().numpy()
    
    if thresh_np.ndim == 1:
        # Global case: (num_bits,)
        for bit_idx in range(len(thresh_np)):
            col_name = f"thresh_{bit_idx}"
            row[col_name] = float(thresh_np[bit_idx])
    
    else:
        # Multi-dimensional case: (..., num_bits)
        # Iterate through all indices
        import numpy as np
        for index in np.ndindex(thresh_np.shape):
            # Create column name from indices: thresh_0_1_2 for index (0,1,2)
            col_name = "thresh_" + "_".join(map(str, index))
            row[col_name] = float(thresh_np[index])
    
    # Write to CSV
    fieldnames = ['step'] + sorted([k for k in row.keys() if k != 'step'])
    
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=fieldnames, lineterminator="\n"
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)



def save_config(config: Dict[str, Any], output_path: Path, filename: str = "config.json"):
    """Save configuration to JSON file."""
    filepath = os.path.join(output_path, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Convert non-serializable objects to strings
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, Path):
            serializable_config[key] = str(value)
        elif hasattr(value, '__dict__'):
            serializable_config[key] = str(value)
        else:
            serializable_config[key] = value

    with open(filepath, 'w') as f:
        json.dump(serializable_config, f, indent=2, default=str)


def load_model_from_checkpoint(model_path: Path, model_class, **model_kwargs):
    """Load a trained model from checkpoint."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Initialize model
    model = model_class(**model_kwargs)

    # Load state dict
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)

    return model


def evaluate_model(model, loader, eval_functions, mode="eval", device="cuda"):
    """Evaluate a model, weighting batch means by the number of examples."""
    orig_mode = model.training
    model.train(mode == "train")

    metric_sums = defaultdict(float)
    example_count = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if mode == "packbits":
                x = torchlogix.PackBitsTensor(x.reshape(x.shape[0], -1).round().bool())

            preds = model(x)
            batch_size = y.shape[0]
            example_count += batch_size

            for name, fn in eval_functions.items():
                batch_mean = fn(preds, y).to(torch.float32).mean().item()
                metric_sums[name] += batch_mean * batch_size

    model.train(orig_mode)

    if example_count == 0:
        raise ValueError("Cannot evaluate an empty data loader")
    return {name: total / example_count for name, total in metric_sums.items()}
