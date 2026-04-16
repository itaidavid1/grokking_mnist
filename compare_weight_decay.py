"""Compare pre-training with different weight decay values.

This script demonstrates the effect of L2 regularization (weight decay) on
pre-training performance.
"""
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from grokking_mnist.config import GrokConfig
    from grokking_mnist.train import train
    from grokking_mnist.data import load_geometric_shapes
else:
    from .config import GrokConfig
    from .train import train
    from .data import load_geometric_shapes


def run_comparison():
    """Run pre-training with different weight decay values for comparison."""
    
    weight_decay_values = [0.0, 0.01, 0.05, 0.1, 0.2]
    
    print("=" * 70)
    print("Comparing Pre-training with Different Weight Decay (L2 Regularization)")
    print("=" * 70)
    print()
    
    for wd in weight_decay_values:
        print(f"\n{'=' * 70}")
        print(f"Running with weight_decay = {wd}")
        print(f"{'=' * 70}\n")
        
        config = GrokConfig(
            seed=42,
            optimization_steps=2000,
            batch_size=50,
            lr=1e-3,
            weight_decay=wd,
            optimizer="AdamW",
            loss_function="MSE",
            depth=3,
            width=200,
            activation="ReLU",
            initialization_scale=8.0,
            filter="none",
            label=f"compare_wd{wd:.2f}".replace(".", ""),
            results_dir="./results/weight_decay_comparison",
        )
        
        # Load shapes dataset
        train_set, test_set = load_geometric_shapes(config.seed, num_samples_per_shape=100)
        
        # Train
        results, model = train(config, train_set=train_set, test_set=test_set, num_classes=4)
        
        final_train_acc = results['train_acc'][-1]
        final_test_acc = results['test_acc'][-1]
        
        print(f"\nFinal Results for weight_decay={wd}:")
        print(f"  Train Accuracy: {final_train_acc*100:.2f}%")
        print(f"  Test Accuracy:  {final_test_acc*100:.2f}%")
        print(f"  Generalization Gap: {(final_train_acc - final_test_acc)*100:.2f}%")
    
    print("\n" + "=" * 70)
    print("Comparison Complete!")
    print("=" * 70)
    print(f"\nResults saved to: ./results/weight_decay_comparison/")
    print("\nKey Insights:")
    print("  - Higher weight decay typically reduces overfitting")
    print("  - Lower generalization gap suggests better feature learning")
    print("  - Optimal value depends on dataset size and model capacity")


if __name__ == "__main__":
    run_comparison()
