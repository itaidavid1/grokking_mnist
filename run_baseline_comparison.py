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
    from grokking_mnist.grokking_detector import detect_grokking, print_grokking_report
else:
    from .config import GrokConfig
    from .train import train
    from .data import load_mnist
    from .grokking_detector import detect_grokking, print_grokking_report


def run_baseline_experiment(train_points=1000, optimization_steps=20000, base_results_dir="./results/baseline"):
    """Run baseline MNIST training WITHOUT Grokfast.
    
    Args:
        train_points: Number of MNIST training samples
        optimization_steps: Number of training steps
        base_results_dir: Directory to save results
        
    Returns:
        Dictionary with results and grokking info
    """
    print("\n" + "=" * 80)
    print("BASELINE EXPERIMENT (NO GROKFAST)")
    print("=" * 80)
    print(f"MNIST train points: {train_points}")
    print(f"Optimization steps: {optimization_steps}")
    print(f"Grokfast: DISABLED\n")
    
    config = GrokConfig(
        seed=42,
        optimization_steps=optimization_steps,
        batch_size=50,
        lr=1e-3,
        weight_decay=0.0,  # No weight decay for baseline
        optimizer="AdamW",
        loss_function="MSE",
        depth=3,
        width=200,
        activation="ReLU",
        initialization_scale=8.0,
        filter="none",  # NO Grokfast
        label="baseline",
        results_dir=base_results_dir,
        train_points=train_points,
    )
    
    # Load MNIST
    train_set, test_set = load_mnist(config)
    
    # Train
    results, model = train(config, train_set=train_set, test_set=test_set, num_classes=10)
    
    # Detect grokking
    grokking_info = detect_grokking(
        results['log_steps'],
        results['train_acc'],
        results['test_acc']
    )
    print_grokking_report(grokking_info, label="Baseline")
    
    return {
        'experiment': 'baseline',
        'filter': 'none',
        'final_train_acc': results['train_acc'][-1],
        'final_test_acc': results['test_acc'][-1],
        'final_train_loss': results['train_loss'][-1],
        'final_test_loss': results['test_loss'][-1],
        'grokking_info': grokking_info
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
    train_set, test_set = load_mnist(config)
    
    # Train
    results, model = train(config, train_set=train_set, test_set=test_set, num_classes=10)
    
    # Detect grokking
    grokking_info = detect_grokking(
        results['log_steps'],
        results['train_acc'],
        results['test_acc']
    )
    print_grokking_report(grokking_info, label="Grokfast")
    
    return {
        'experiment': 'grokfast',
        'filter': 'ema',
        'final_train_acc': results['train_acc'][-1],
        'final_test_acc': results['test_acc'][-1],
        'final_train_loss': results['train_loss'][-1],
        'final_test_loss': results['test_loss'][-1],
        'grokking_info': grokking_info
    }


def print_comparison_table(results):
    """Print comparison table of baseline vs grokfast."""
    print("\n" + "=" * 100)
    print("BASELINE vs GROKFAST COMPARISON")
    print("=" * 100)
    print(f"\n{'Experiment':<15} {'Filter':<10} {'Train Acc':<12} {'Test Acc':<12} {'Gap':<10} {'Grokking':<10}")
    print("-" * 100)
    
    for result in results:
        experiment = result['experiment']
        filter_type = result['filter']
        train_acc = result['final_train_acc'] * 100
        test_acc = result['final_test_acc'] * 100
        gap = train_acc - test_acc
        grokked = "YES" if result['grokking_info']['grokking_detected'] else "NO"
        
        print(f"{experiment:<15} {filter_type:<10} {train_acc:<12.2f} {test_acc:<12.2f} {gap:<10.2f} {grokked:<10}")
    
    print("=" * 100)
    
    # Print difference
    if len(results) == 2:
        baseline = results[0]
        grokfast = results[1]
        
        test_acc_diff = (grokfast['final_test_acc'] - baseline['final_test_acc']) * 100
        
        print(f"\nGrokfast Improvement:")
        print(f"  Test Accuracy Gain: {test_acc_diff:+.2f}%")
        
        if baseline['grokking_info']['grokking_detected'] or grokfast['grokking_info']['grokking_detected']:
            print(f"  Grokking Effect:")
            if grokfast['grokking_info']['grokking_detected']:
                print(f"    - Grokfast: Grokking at step {grokfast['grokking_info']['grokking_step']}")
            else:
                print(f"    - Grokfast: No grokking detected")
            
            if baseline['grokking_info']['grokking_detected']:
                print(f"    - Baseline: Grokking at step {baseline['grokking_info']['grokking_step']}")
            else:
                print(f"    - Baseline: No grokking detected")


def save_summary(results, base_results_dir):
    """Save comparison summary to JSON file."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'experiments': results,
    }
    
    summary_path = f"{base_results_dir}/comparison_summary.json"
    os.makedirs(base_results_dir, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nComparison summary saved to: {summary_path}")


def main():
    """Run baseline vs grokfast comparison."""
    # Configuration
    TRAIN_POINTS = 1000
    OPTIMIZATION_STEPS = 20000
    BASE_RESULTS_DIR = "./results/baseline_vs_grokfast"
    
    print("=" * 100)
    print("BASELINE vs GROKFAST COMPARISON EXPERIMENT")
    print("=" * 100)
    print(f"\nConfiguration:")
    print(f"  MNIST train points: {TRAIN_POINTS}")
    print(f"  Optimization steps: {OPTIMIZATION_STEPS}")
    print(f"  Total experiments: 2 (Baseline + Grokfast)")
    print(f"  Results directory: {BASE_RESULTS_DIR}\n")
    
    results = []
    
    # Run baseline
    baseline_result = run_baseline_experiment(
        train_points=TRAIN_POINTS,
        optimization_steps=OPTIMIZATION_STEPS,
        base_results_dir=f"{BASE_RESULTS_DIR}/baseline"
    )
    results.append(baseline_result)
    
    # Run grokfast
    grokfast_result = run_grokfast_experiment(
        train_points=TRAIN_POINTS,
        optimization_steps=OPTIMIZATION_STEPS,
        base_results_dir=f"{BASE_RESULTS_DIR}/grokfast"
    )
    results.append(grokfast_result)
    
    # Print comparison
    print_comparison_table(results)
    
    # Save summary
    save_summary(results, BASE_RESULTS_DIR)
    
    print("\n" + "=" * 100)
    print("COMPARISON COMPLETE!")
    print("=" * 100)
    print(f"\nAll results saved to: {BASE_RESULTS_DIR}/")
    print(f"  - Baseline results: {BASE_RESULTS_DIR}/baseline/")
    print(f"  - Grokfast results: {BASE_RESULTS_DIR}/grokfast/")
    print(f"  - Comparison summary: {BASE_RESULTS_DIR}/comparison_summary.json")


if __name__ == "__main__":
    main()
