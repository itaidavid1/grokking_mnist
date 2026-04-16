"""Comparison script: Baseline vs Grokfast on direct MNIST training.

This script runs two experiments for fair comparison:
1. Baseline: Direct MNIST training WITHOUT Grokfast
2. Grokfast: Direct MNIST training WITH Grokfast (EMA filter)

Both use identical settings (dataset size, steps, architecture) to isolate
the effect of Grokfast gradient filtering.
"""
import os
import sys
import json
import torch
from datetime import datetime

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from grokking_mnist.config import GrokConfig
    from grokking_mnist.train import train
    from grokking_mnist.data import load_mnist
else:
    from .config import GrokConfig
    from .train import train
    from .data import load_mnist


def run_baseline_experiment(
    weight_decay,
    train_points=1000,
    optimization_steps=20000,
    base_results_dir="./results/baseline",
):
    """Run baseline MNIST training WITHOUT Grokfast."""
    print("\n" + "=" * 80)
    print("BASELINE EXPERIMENT (NO GROKFAST)")
    print("=" * 80)
    print(f"MNIST train points: {train_points}")
    print(f"Optimization steps: {optimization_steps}")
    print(f"Weight decay: {weight_decay}")

    config = GrokConfig(
        seed=42,
        optimization_steps=optimization_steps,
        batch_size=50,
        lr=1e-3,
        weight_decay=weight_decay,
        optimizer="AdamW",
        loss_function="MSE",
        depth=3,
        width=200,
        activation="ReLU",
        initialization_scale=8.0,
        filter="none",
        label=f"baseline_wd{weight_decay:.6f}".replace(".", ""),
        results_dir=base_results_dir,
        train_points=train_points,
    )

    train_set, test_set = load_mnist(config.download_directory, train_points)
    results, _ = train(config, train_set=train_set, test_set=test_set, num_classes=10)

    return {
        'experiment': 'baseline',
        'filter': 'none',
        'final_train_acc': results['train_acc'][-1],
        'final_test_acc': results['test_acc'][-1],
        'final_train_loss': results['train_loss'][-1],
        'final_test_loss': results['test_loss'][-1],
        'weight_decay': weight_decay,
    }


def run_grokfast_experiment(train_points=1000, optimization_steps=20000, base_results_dir="./results/grokfast"):
    """Run MNIST training WITH Grokfast.
    
    Args:
        train_points: Number of MNIST training samples
        optimization_steps: Number of training steps
        base_results_dir: Directory to save results
        
    Returns:
        Dictionary with results and grokking info
    """
    print("\n" + "=" * 80)
    print("GROKFAST EXPERIMENT (EMA FILTER)")
    print("=" * 80)
    print(f"MNIST train points: {train_points}")
    print(f"Optimization steps: {optimization_steps}")
    print(f"Grokfast: ENABLED (EMA with alpha=0.99, lamb=5.0)\n")
    
    config = GrokConfig(
        seed=42,
        optimization_steps=optimization_steps,
        batch_size=50,
        lr=1e-3,
        weight_decay=0.0,  # No weight decay
        optimizer="AdamW",
        loss_function="MSE",
        depth=3,
        width=200,
        activation="ReLU",
        initialization_scale=8.0,
        filter="ema",  # Enable Grokfast
        alpha=0.99,
        lamb=5.0,
        label="grokfast",
        results_dir=base_results_dir,
        train_points=train_points,
    )
    
    # Load MNIST
    train_set, test_set = load_mnist(config.download_directory, train_points)
    
    # Train
    results, model = train(config, train_set=train_set, test_set=test_set, num_classes=10)
    
    
    return {
        'experiment': 'grokfast',
        'filter': 'ema',
        'final_train_acc': results['train_acc'][-1],
        'final_test_acc': results['test_acc'][-1],
        'final_train_loss': results['train_loss'][-1],
        'final_test_loss': results['test_loss'][-1],
        'weight_decay': 0.0,
    }


def print_comparison_table(results):
    """Print comparison table of baseline vs grokfast."""
    print("\n" + "=" * 100)
    print("BASELINE vs GROKFAST COMPARISON")
    print("=" * 100)
    print(f"\n{'Weight Decay':<14} {'Experiment':<15}  {'Filter':<10} {'Train Acc':<12} {'Test Acc':<12} {'Gap':<10}")
    print("-" * 100)

    for result in results:
        experiment = result["experiment"]
        filter = result["filter"]
        weight_decay = result["weight_decay"]
        train_acc = result["final_train_acc"] * 100
        test_acc = result["final_test_acc"] * 100
        gap = train_acc - test_acc

        print(f"{weight_decay:<14.6f} {experiment:<15} {filter:<10} {train_acc:<12.2f} {test_acc:<12.2f} {gap:<10.2f}")

    print("=" * 100)


def save_summary(results, base_results_dir):
    """Save summary JSON for baseline WD sweep."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "baseline_vs_grokfast",
        "results": results,
    }

    os.makedirs(base_results_dir, exist_ok=True)
    summary_path = f"{base_results_dir}/comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nComparison summary saved to: {summary_path}")


def main():
    """Run baseline-only MNIST weight decay sweep."""
    train_points = 1000
    optimization_steps = 20000
    weight_decay_values = [0.0, 0.1, 0.01, 0.001]
    base_results_dir = "./results/baseline_vs_grokfast"

    print("=" * 100)
    print("BASELINE MNIST WEIGHT DECAY SWEEP")
    print("=" * 100)
    print("\nConfiguration:")
    print(f"  MNIST train points: {train_points}")
    print(f"  Optimization steps: {optimization_steps}")
    print(f"  Weight decay values: {weight_decay_values}")
    print(f"  Total experiments: {len(weight_decay_values) + 1} ")
    print(f"  Results directory: {base_results_dir}\n")

    results = []
    for weight_decay in weight_decay_values:
        baseline_result = run_baseline_experiment(
            weight_decay=weight_decay,
            train_points=train_points,
            optimization_steps=optimization_steps,
            base_results_dir=f"{base_results_dir}/baseline"
        )
        results.append(baseline_result)
    
    # Run grokfast
    grokfast_result = run_grokfast_experiment(
        train_points=train_points,
        optimization_steps=optimization_steps,
        base_results_dir=f"{base_results_dir}/grokfast"
    )
    results.append(grokfast_result)
    
    # Print comparison
    print_comparison_table(results)
    save_summary(results, base_results_dir)

    print("\n" + "=" * 100)
    print("SWEEP COMPLETE!")
    print("=" * 100)
    print(f"\nAll results saved to: {base_results_dir}/")
    print(f"  - Per-WD baseline results: {base_results_dir}/wd_*/")
    print(f"  - Comparison summary: {base_results_dir}/comparison_summary.json")


if __name__ == "__main__":
    main()
