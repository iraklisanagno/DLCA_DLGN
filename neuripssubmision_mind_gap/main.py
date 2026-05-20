import os
import sys
import signal
import json
import yaml
import argparse
import time
import random

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from functional import bin_op_s, get_unique_connections, GradFactor
from packbitstensor import PackBitsTensor
from layers import LogicLayer, GroupSum

# Limit CPU threads to avoid oversubscription when using multiple workers
torch.set_num_threads(1)

BITS_TO_TORCH_FLOATING_POINT_TYPE = {
    16: torch.float16,
    32: torch.float32,
    64: torch.float64
}

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

########################################
# New function to load a checkpoint
########################################
def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model/optimizer state and return (start_iter, best_acc)."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["w_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_iter = checkpoint["iteration"]
    best_acc = checkpoint["best_acc"]
    print(f"[INFO] Loaded checkpoint from iteration {start_iter}, best_acc so far: {best_acc:.4f}")
    return start_iter, best_acc

########################################
# Modified save_results to optionally save checkpoints
########################################
def save_results(config, results, model=None, optimizer=None, iteration=None, best_acc=None, best_model=False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config.get("experiment_name", "experiment")
    output_dir = Path("experiments/results") / f"{exp_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config and metrics
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save model weights if provided
    if model is not None:
        torch.save(model.state_dict(), output_dir / "model.pt")

    # Optionally save full checkpoint
    if model is not None and optimizer is not None and iteration is not None and best_acc is not None:
        checkpoint = {
            'w_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': iteration,
            'best_acc': best_acc,
            'config': config
        }
        ckpt_name = "best_checkpoint.pt" if best_model else "checkpoint.pt"
        torch.save(checkpoint, output_dir / ckpt_name)

    print(f"[INFO] Results saved to {output_dir}")


def load_n(loader, n):
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i == n:
                break


def train_step(model, x, y, loss_fn, optimizer):
    out = model(x)
    loss = loss_fn(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def eval_mode(model, loader, mode):
    orig = model.training
    with torch.no_grad():
        model.train(mode=mode)
        accs = []
        for x, y in loader:
            pred = model(x.to('cuda').round()).argmax(-1)
            accs.append((pred == y.to('cuda')).float().mean().item())
        res = float(np.mean(accs))
    model.train(mode=orig)
    return res


def packbits_eval(model, loader):
    orig = model.training
    with torch.no_grad():
        model.eval()
        accs = []
        for x, y in loader:
            x_pb = PackBitsTensor(x.to('cuda').reshape(x.shape[0], -1).round().bool())
            pred = model(x_pb).argmax(-1)
            accs.append((pred == y.to('cuda')).float().mean().item())
        res = float(np.mean(accs))
    model.train(mode=orig)
    return res


def run_experiment(config):
    torch.set_num_threads(1)
    # Hyperparams
    batch_size        = config["batch_size"]
    num_iterations    = config["num_iterations"]
    training_bit_count= config["training_bit_count"]

    # Seeding
    seed = config.get("seed", None)
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"[INFO] Using fixed seed: {seed}")

    # Dataset setup
    dataset_name = config.get("dataset", "CIFAR10").upper()
    if dataset_name == "CIFAR10":
        binarize = lambda x: torch.cat([(x > (i+1)/32).float() for i in range(31)], dim=0)
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(binarize)
        ])
        train_set = torchvision.datasets.CIFAR10('./data-cifar', train=True, download=True, transform=transforms)
        test_set  = torchvision.datasets.CIFAR10('./data-cifar', train=False, download=True, transform=transforms)
        in_dim    = 3 * 32 * 32 * 31
    elif dataset_name == "MNIST":
        transforms = torchvision.transforms.ToTensor()
        train_set = torchvision.datasets.MNIST('./data-mnist', train=True, download=True, transform=transforms)
        test_set  = torchvision.datasets.MNIST('./data-mnist', train=False, download=True, transform=transforms)
        in_dim    = 28 * 28
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Optional label shuffling
    if config.get("shuffle_train_labels", False):
        print("[INFO] Shuffling training labels")
        targets = getattr(train_set, 'targets', getattr(train_set, 'labels', None))
        idx = torch.randperm(len(targets))
        if isinstance(targets, torch.Tensor):
            train_set.targets = targets[idx]
        else:
            train_set.targets = torch.tensor(targets)[idx].tolist()

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, pin_memory=True,
                                               drop_last=True, num_workers=4)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=batch_size,
                                               shuffle=False, pin_memory=True,
                                               drop_last=False, num_workers=4)

    # Model
    llkw = dict(grad_factor=config["grad_factor"], connections='unique')
    layers = [torch.nn.Flatten(),
              LogicLayer(in_dim=in_dim, out_dim=config["num_neurons"], stochastic=config["stochastic"],
                         gumbel_tau=config.get("gumbel_tau",1.0), **llkw)]
    for _ in range(config["num_layers"] - 1):
        layers.append(
            LogicLayer(in_dim=config["num_neurons"], out_dim=config["num_neurons"],
                       stochastic=config["stochastic"], gumbel_tau=config.get("gumbel_tau",1.0), **llkw)
        )
    layers.append(GroupSum(k=10, tau=config["tau"]))
    model = torch.nn.Sequential(*layers).to('cuda')

    loss_fn   = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Checkpoint resume
    resume_checkpoint = config.get("resume_checkpoint")
    start_iter, best_acc = 0, 0.0
    if resume_checkpoint and Path(resume_checkpoint).exists():
        print(f"[INFO] Resuming from {resume_checkpoint}")
        start_iter, best_acc = load_checkpoint(model, optimizer, resume_checkpoint)

    # Setup SIGTERM handler for graceful checkpoint
    def _term_handler(signum, frame):
        print("[INFO] SIGTERM caught, saving checkpoint…")
        save_results(config, {'iteration': i, 'duration_sec': time.time()-start_time},
                     model=model, optimizer=optimizer,
                     iteration=i, best_acc=best_acc, best_model=False)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _term_handler)

    # Periodic save settings
    SAVE_INTERVAL_ITERS = config.get("save_interval_iters", 5000)
    SAVE_INTERVAL_SECS  = config.get("save_interval_secs", 3600)
    last_save_time = time.time()

    start_time = time.time()
    # Training loop
    for i, (x, y) in tqdm(enumerate(load_n(train_loader, num_iterations)), total=num_iterations):
        if i < start_iter:
            continue

        x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[training_bit_count]).to('cuda')
        y = y.to('cuda')

        loss = train_step(model, x, y, loss_fn, optimizer)

        # Periodic checkpoint by iteration
        if (i+1) % SAVE_INTERVAL_ITERS == 0:
            save_results(config, {'iteration': i+1, 'duration_sec': time.time()-start_time},
                         model=model, optimizer=optimizer,
                         iteration=i+1, best_acc=best_acc, best_model=False)

        # Periodic checkpoint by time
        if time.time() - last_save_time > SAVE_INTERVAL_SECS:
            save_results(config, {'iteration': i+1, 'duration_sec': time.time()-start_time},
                         model=model, optimizer=optimizer,
                         iteration=i+1, best_acc=best_acc, best_model=False)
            last_save_time = time.time()

        # Evaluation & best-model saving every 2000 iters
        if (i + 1) % 2000 == 0:
            metrics = {
                'train_acc_eval_mode': eval_mode(model, train_loader, mode=False),
                'train_acc_train_mode': eval_mode(model, train_loader, mode=True),
                'test_acc_eval_mode': eval_mode(model, test_loader, mode=False),
                'test_acc_train_mode': eval_mode(model, test_loader, mode=True),
                'test_acc_packbits': packbits_eval(model, test_loader),
                'stochastic': config["stochastic"],
                'iteration': i+1,
                'duration_sec': time.time() - start_time
            }
            current_test_acc = metrics['test_acc_eval_mode']
            is_best = current_test_acc > best_acc
            if is_best:
                best_acc = current_test_acc
            save_results(config, metrics, model=model, optimizer=optimizer,
                         iteration=i+1, best_acc=best_acc, best_model=is_best)

    # Final save
    if config.get("save_final_model", False):
        final_metrics = {
            'train_acc_eval_mode': eval_mode(model, train_loader, mode=False),
            'train_acc_train_mode': eval_mode(model, train_loader, mode=True),
            'test_acc_eval_mode': eval_mode(model, test_loader, mode=False),
            'test_acc_train_mode': eval_mode(model, test_loader, mode=True),
            'test_acc_packbits': packbits_eval(model, test_loader),
            'stochastic': config["stochastic"],
            'iteration': num_iterations,
            'duration_sec': time.time() - start_time
        }
        is_best = final_metrics['test_acc_eval_mode'] > best_acc
        if is_best:
            best_acc = final_metrics['test_acc_eval_mode']
        save_results(config, final_metrics, model=model, optimizer=optimizer,
                     iteration=num_iterations, best_acc=best_acc, best_model=is_best)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_experiment(cfg)