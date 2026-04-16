import math
import os
import random
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import (
    LOSS_FUNCTION_DICT,
    OPTIMIZER_DICT,
    GrokConfig,
)
from .data import load_mnist
from .grokfast import gradfilter_ema, gradfilter_ma
from .model import build_mlp


# ── helpers ──────────────────────────────────────────────────────────────────

def _cycle(iterable):
    while True:
        for x in iterable:
            yield x


def compute_accuracy(network, loader, device, max_batches=None):
    with torch.no_grad():
        correct = 0
        total = 0
        batch_iter = islice(loader, max_batches) if max_batches else loader
        for x, labels in batch_iter:
            logits = network(x.to(device))
            predicted = torch.argmax(logits, dim=1)
            correct += torch.sum(predicted == labels.to(device))
            total += x.size(0)
    return (correct / total).item()


def compute_loss(network, loader, loss_function, device, num_classes=10, max_batches=None):
    with torch.no_grad():
        loss_fn = LOSS_FUNCTION_DICT[loss_function](reduction="sum")
        one_hots = torch.eye(num_classes, num_classes).to(device)
        total = 0.0
        points = 0
        batch_iter = islice(loader, max_batches) if max_batches else loader
        for x, labels in batch_iter:
            y = network(x.to(device))
            if loss_function == "CrossEntropy":
                total += loss_fn(y, labels.to(device)).item()
            elif loss_function == "MSE":
                total += loss_fn(y, one_hots[labels]).item()
            points += len(labels)
    return total / points


# ── plotting ─────────────────────────────────────────────────────────────────

def _save_plots(log_steps, train_acc, test_acc, train_loss, test_loss,
                 results_dir, label, optimization_steps):
    os.makedirs(results_dir, exist_ok=True)

    steps = np.array(log_steps)

    plt.figure()
    plt.plot(steps, train_acc, label="train")
    plt.plot(steps, test_acc, label="test")
    plt.legend()
    plt.title("Accuracy")
    plt.xlabel("Optimization Steps")
    plt.ylabel("Accuracy")
    plt.xlim(0, optimization_steps)
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"acc_{label}.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(steps, train_loss, label="train")
    plt.plot(steps, test_loss, label="test")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Optimization Steps")
    plt.ylabel("Loss")
    plt.xlim(0, optimization_steps)
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"loss_{label}.png"), dpi=150)
    plt.close()


# ── main training entry-point ────────────────────────────────────────────────

def train(config: GrokConfig, model=None, train_set=None, test_set=None, num_classes=10):
    """Train a model on the given dataset.
    
    Args:
        config: Training configuration
        train_set: Training dataset (if None, loads MNIST)
        test_set: Test dataset (if None, loads MNIST)
        num_classes: Number of output classes
    """
    # --- reproducibility ---
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    torch.set_default_dtype(dtype)

    log_freq = math.ceil(config.optimization_steps / config.log_freq_divisor)

    # --- data ---
    if train_set is None or test_set is None:
        train_set, test_set = load_mnist(config)
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
    )
    eval_batch = 200
    train_eval_loader = torch.utils.data.DataLoader(
        train_set, batch_size=eval_batch, shuffle=False,
    )
    test_eval_loader = torch.utils.data.DataLoader(
        test_set, batch_size=eval_batch, shuffle=False,
    )
    test_max_batches = 10  # cap test eval at ~2000 samples to stay fast

    # --- model ---
    if model is None:
        mlp = build_mlp(config, device, num_classes=num_classes)
    else:
        mlp = model
    nparams = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    print(f"Number of parameters: {nparams}")

    # --- optimizer ---
    assert config.optimizer in OPTIMIZER_DICT, (
        f"Unsupported optimizer: {config.optimizer}"
    )
    optimizer = OPTIMIZER_DICT[config.optimizer](
        mlp.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )

    # --- loss ---
    assert config.loss_function in LOSS_FUNCTION_DICT, (
        f"Unsupported loss: {config.loss_function}"
    )
    loss_fn = LOSS_FUNCTION_DICT[config.loss_function]()
    one_hots = torch.eye(num_classes, num_classes).to(device)

    # --- metric storage ---
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    norms, last_layer_norms, log_steps = [], [], []

    grads = None
    steps = 0

    with tqdm(total=config.optimization_steps, dynamic_ncols=True) as pbar:
        for x, labels in islice(_cycle(train_loader), config.optimization_steps):

            # --- periodic evaluation ---
            do_log = (
                steps == 0
                or (steps < 150 and steps % 10 == 0)
                or steps % log_freq == 0
            )
            if do_log:
                train_loss = compute_loss(mlp, train_eval_loader, config.loss_function, device, num_classes=num_classes)
                train_acc = compute_accuracy(mlp, train_eval_loader, device)
                test_loss = compute_loss(mlp, test_eval_loader, config.loss_function, device, num_classes=num_classes, max_batches=test_max_batches)
                test_acc = compute_accuracy(mlp, test_eval_loader, device, max_batches=test_max_batches)

                train_losses.append(train_loss)
                train_accuracies.append(train_acc)
                test_losses.append(test_loss)
                test_accuracies.append(test_acc)
                log_steps.append(steps)

                norm = sum(p.data.norm().item() ** 2 for p in mlp.parameters()) ** 0.5
                norms.append(norm)
                last_layer_norm = list(mlp.parameters())[-1].data.norm().item()
                last_layer_norms.append(last_layer_norm)

                pbar.set_description(
                    f"trL={train_loss:.4f} teL={test_loss:.4f} "
                    f"trA={train_acc * 100:.1f}% teA={test_acc * 100:.1f}%"
                )

            # --- forward ---
            y = mlp(x.to(device))
            if config.loss_function == "CrossEntropy":
                loss = loss_fn(y, labels.to(device))
            elif config.loss_function == "MSE":
                loss = loss_fn(y, one_hots[labels])

            # --- backward + grokfast filter ---
            optimizer.zero_grad()
            loss.backward()

            if config.filter == "none":
                pass
            elif config.filter == "ma":
                grads = gradfilter_ma(
                    mlp, grads=grads,
                    window_size=config.window_size, lamb=config.lamb,
                )
            elif config.filter == "ema":
                grads = gradfilter_ema(
                    mlp, grads=grads,
                    alpha=config.alpha, lamb=config.lamb,
                )
            else:
                raise ValueError(f"Invalid gradient filter type '{config.filter}'")

            optimizer.step()
            steps += 1
            pbar.update(1)

    # --- save results ---
    os.makedirs(config.results_dir, exist_ok=True)

    _save_plots(
        log_steps, train_accuracies, test_accuracies,
        train_losses, test_losses, config.results_dir, config.label,
        config.optimization_steps,
    )

    results = {
        "log_steps": log_steps,
        "train_loss": train_losses,
        "test_loss": test_losses,
        "train_acc": train_accuracies,
        "test_acc": test_accuracies,
        "norms": norms,
        "last_layer_norms": last_layer_norms,
    }
    torch.save(results, os.path.join(config.results_dir, f"res_{config.label}.pt"))

    print(f"\nResults saved to {config.results_dir}/res_{config.label}.pt")
    print(f"Plots saved to {config.results_dir}/acc_{config.label}.png and loss_{config.label}.png")
    return results, mlp
