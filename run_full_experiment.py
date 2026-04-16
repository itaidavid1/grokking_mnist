"""Comprehensive pre-training and fine-tuning experiment.

This script runs a full experiment matrix:
- Pre-train on geometric shapes with 5 different weight decay values
- Save checkpoints from each pre-training run
- Fine-tune each checkpoint on MNIST with 3 different weight decay values
- Total: 5 × 3 = 15 experiments

Results are saved in a structured directory for easy comparison.
"""
import os
import sys
import json
import torch
from datetime import datetime
from collections import OrderedDict

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from grokking_mnist.config import GrokConfig
    from grokking_mnist.train import train
    from grokking_mnist.data import load_geometric_shapes, load_mnist
    from grokking_mnist.model import build_mlp
else:
    from .config import GrokConfig
    from .train import train
    from .data import load_geometric_shapes, load_mnist
    from .model import build_mlp


def save_checkpoint(model, config, filepath):
    """Save model checkpoint with configuration."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': vars(config),
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to: {filepath}")


def load_checkpoint_weights(model, filepath):
    """Load model weights from checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded weights from: {filepath}")
    return model


def build_finetune_wd_combinations(pretrain_wd_values, finetune_fractions, explicit_mnist_wd_values):
    """Build de-duplicated pretrain/fine-tune WD combinations.

    Rules:
      - pretrain_wd > 0: use ratio-derived values (pretrain_wd * fraction)
      - pretrain_wd == 0: force one no-WD run (finetune_wd=0)
      - pretrain_wd == 0: also test explicit MNIST WD values > 0
    """
    combos = OrderedDict()

    for pretrain_wd in pretrain_wd_values:
        if pretrain_wd > 0:
            for fraction in finetune_fractions:
                finetune_wd = pretrain_wd * fraction
                combo_key = (pretrain_wd, finetune_wd)
                combos[combo_key] = {
                    "pretrain_wd": pretrain_wd,
                    "finetune_wd": finetune_wd,
                    "finetune_fraction": fraction,
                    "wd_source": "ratio",
                }
        else:
            combo_key = (0.0, 0.0)
            combos[combo_key] = {
                "pretrain_wd": 0.0,
                "finetune_wd": 0.0,
                "finetune_fraction": None,
                "wd_source": "forced_zero",
            }

            for explicit_wd in explicit_mnist_wd_values:
                if explicit_wd <= 0:
                    continue
                combo_key = (0.0, explicit_wd)
                combos[combo_key] = {
                    "pretrain_wd": 0.0,
                    "finetune_wd": explicit_wd,
                    "finetune_fraction": None,
                    "wd_source": "mnist_explicit",
                }

    return list(combos.values())


def run_pretraining_experiments(
    pretrain_wd_values,
    samples_per_shape=250,
    optimization_steps=1000,
    base_results_dir="./results/full_experiment"
):
    """Run pre-training experiments with different weight decay values.
    
    Returns:
        List of checkpoint paths
    """
    print("=" * 80)
    print("PHASE 1: PRE-TRAINING ON GEOMETRIC SHAPES")
    print("=" * 80)
    print(f"\nWeight decay values: {pretrain_wd_values}")
    print(f"Samples per shape: {samples_per_shape}")
    print(f"Optimization steps: {optimization_steps}\n")
    
    checkpoint_paths = []
    pretrain_results = []
    
    for i, wd in enumerate(pretrain_wd_values, 1):
        print(f"\n{'=' * 80}")
        print(f"Pre-training {i}/{len(pretrain_wd_values)}: weight_decay = {wd}")
        print(f"{'=' * 80}\n")
        
        config = GrokConfig(
            seed=42,
            optimization_steps=optimization_steps,
            batch_size=50,
            lr=1e-3,
            weight_decay=wd,
            optimizer="AdamW",
            loss_function="MSE",
            depth=3,
            width=200,
            activation="ReLU",
            initialization_scale=8.0,
            filter="none",  # NO Grokfast - test pure pre-training effect
            label=f"pretrain_wd{wd:.4f}".replace(".", ""),
            results_dir=f"{base_results_dir}/pretraining",
        )
        
        # Load shapes dataset
        train_set, test_set = load_geometric_shapes(config.seed, num_samples_per_shape=samples_per_shape)
        
        # Train
        results, model = train(config, train_set=train_set, test_set=test_set, num_classes=4)
        
        # Save checkpoint
        checkpoint_path = f"{base_results_dir}/checkpoints/pretrain_wd{wd:.4f}.pt".replace(".", "")
        save_checkpoint(model, config, checkpoint_path)
        checkpoint_paths.append((wd, checkpoint_path))
        
        # Store results
        pretrain_results.append({
            'pretrain_wd': wd,
            'final_train_acc': results['train_acc'][-1],
            'final_test_acc': results['test_acc'][-1],
            'checkpoint': checkpoint_path
        })
    
    return checkpoint_paths, pretrain_results


def run_finetuning_experiments(
    checkpoint_paths,
    finetune_combinations,
    train_points=1000,
    optimization_steps=20000,
    base_results_dir="./results/full_experiment"
):
    """Run fine-tuning experiments on MNIST for each checkpoint.
    
    Args:
        checkpoint_paths: List of (pretrain_wd, checkpoint_path) tuples
        finetune_combinations: List of WD combination dictionaries
        train_points: Number of MNIST training samples
        optimization_steps: Training steps for fine-tuning
        base_results_dir: Base directory for results
        
    Returns:
        List of all experiment results
    """
    print("\n" + "=" * 80)
    print("PHASE 2: FINE-TUNING ON MNIST")
    print("=" * 80)
    print(f"\nTotal fine-tuning combinations: {len(finetune_combinations)}")
    print(f"MNIST train points: {train_points}")
    print(f"Optimization steps: {optimization_steps}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = []
    
    checkpoint_map = {pretrain_wd: checkpoint_path for pretrain_wd, checkpoint_path in checkpoint_paths}
    total_experiments = len(finetune_combinations)
    exp_count = 0

    for combo in finetune_combinations:
        pretrain_wd = combo["pretrain_wd"]
        finetune_wd = combo["finetune_wd"]
        finetune_fraction = combo["finetune_fraction"]
        wd_source = combo["wd_source"]
        checkpoint_path = checkpoint_map[pretrain_wd]

        exp_count += 1

        print(f"\n{'=' * 80}")
        print(f"Experiment {exp_count}/{total_experiments}")
        if finetune_fraction is None:
            print(f"Pre-train WD: {pretrain_wd} -> Fine-tune WD: {finetune_wd} ({wd_source})")
        else:
            print(f"Pre-train WD: {pretrain_wd} -> Fine-tune WD: {finetune_wd} (1/{int(1/finetune_fraction) if finetune_fraction != 0 else 0})")
            if finetune_fraction == 0:
                print(f"  NOTE: finetune_fraction is 0, so fine-tuning will be done with no weight decay")
        print(f"{'=' * 80}\n")

        config = GrokConfig(
            seed=42,
            optimization_steps=optimization_steps,
            batch_size=50,
            lr=1e-3, 
            weight_decay=finetune_wd,
            optimizer="AdamW",
            loss_function="MSE",
            depth=3,
            width=200,
            activation="ReLU",
            initialization_scale=8.0,  # Don't rescale when loading pretrained
            filter="none",  # NO Grokfast - test pure pre-training effect
            label=f"ft_pre{pretrain_wd:.4f}_ft{finetune_wd:.6f}_{wd_source}".replace(".", ""),
            results_dir=f"{base_results_dir}/finetuning",
            train_points=train_points,
        )

        # Load MNIST
        mnist_train, mnist_test = load_mnist(config.download_directory, train_points)

        # Create model and load pretrained weights
        model = build_mlp(config, device, num_classes=10)

        # Load pretrained weights (only for matching layers)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        pretrained_state = checkpoint['model_state_dict']

        # Load weights layer by layer, skip the final layer (different output size)
        model_state = model.state_dict()
        pretrained_filtered = {}

        for name, param in pretrained_state.items():
            if name in model_state and param.shape == model_state[name].shape:
                pretrained_filtered[name] = param
                print(f"  Loaded pretrained layer: {name}")
            else:
                print(f"  Skipped layer (shape mismatch or not found): {name}")

        model.load_state_dict(pretrained_filtered, strict=False)
        print(f"Loaded {len(pretrained_filtered)} pretrained layers\n")

        # Fine-tune on MNIST
        results, model = train(config, model=model, train_set=mnist_train, test_set=mnist_test, num_classes=10)

        # Store results
        all_results.append({
            'pretrain_wd': pretrain_wd,
            'finetune_wd': finetune_wd,
            'finetune_fraction': finetune_fraction,
            'wd_source': wd_source,
            'final_train_acc': results['train_acc'][-1],
            'final_test_acc': results['test_acc'][-1],
            'final_train_loss': results['train_loss'][-1],
            'final_test_loss': results['test_loss'][-1],
        })
    
    return all_results


def save_summary(pretrain_results, finetune_results, base_results_dir):
    """Save experiment summary to JSON file."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'pretraining': pretrain_results,
        'finetuning': finetune_results,
    }
    
    summary_path = f"{base_results_dir}/experiment_summary.json"
    os.makedirs(base_results_dir, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nExperiment summary saved to: {summary_path}")


def print_results_table(finetune_results):
    """Print a formatted table of all results."""
    print("\n" + "=" * 100)
    print("FINAL RESULTS SUMMARY")
    print("=" * 100)
    print(f"\n{'Pretrain WD':<12} {'Finetune WD':<12} {'FT Setting':<14} {'WD Source':<14} "
          f"{'Train Acc':<12} {'Test Acc':<12} {'Gap':<10}")
    print("-" * 100)
    
    for result in finetune_results:
        pretrain_wd = result['pretrain_wd']
        finetune_wd = result['finetune_wd']
        fraction = result['finetune_fraction']
        wd_source = result['wd_source']
        train_acc = result['final_train_acc'] * 100
        test_acc = result['final_test_acc'] * 100
        gap = train_acc - test_acc

        if fraction is None:
            ft_setting = "explicit"
        else:
            ft_setting = f"1/{int(1/fraction) if fraction != 0 else 0}"

        print(f"{pretrain_wd:<12.4f} {finetune_wd:<12.6f} {ft_setting:<14} {wd_source:<14} "
              f"{train_acc:<12.2f} {test_acc:<12.2f} {gap:<10.2f}")
    
    print("=" * 100)


def main():
    """Run the full pre-training + fine-tuning experiment (NO Grokfast)."""
    # Configuration
    PRETRAIN_WD_VALUES = [0.0, 0.01, 0.05, 0.1, 1.0]  # 5 different weight decays for pre-training
    FINETUNE_FRACTIONS = [0, 1/10, 1/100, 1/1000]  # 4 fractions of pretrain WD
    FINETUNE_MNIST_WD_VALUES = [0.0, 0.1, 0.01, 0.001]  # explicit MNIST WD sweep for pretrain_wd=0
    
    BASE_RESULTS_DIR = "./results/pretrain_finetune"
    PRETRAIN_SAMPLES_PER_SHAPE = 250
    PRETRAIN_STEPS = 10
    FINETUNE_TRAIN_POINTS = 1000
    FINETUNE_STEPS = 10
    
    print("=" * 100)
    print("PRE-TRAINING + FINE-TUNING EXPERIMENT (NO GROKFAST)")
    print("=" * 100)
    print(f"\nConfiguration:")
    print(f"  Grokfast: DISABLED (testing pure pre-training effect)")
    print(f"  Pre-training weight decays: {PRETRAIN_WD_VALUES} (note: higher WD = stronger regularization)")
    print(f"  Fine-tuning fractions: {[f'1/{int(1/f) if f != 0 else 0}' for f in FINETUNE_FRACTIONS]}")
    print(f"  Explicit MNIST WD sweep (used for pretrain_wd=0): {FINETUNE_MNIST_WD_VALUES}")

    finetune_combinations = build_finetune_wd_combinations(
        PRETRAIN_WD_VALUES,
        FINETUNE_FRACTIONS,
        FINETUNE_MNIST_WD_VALUES,
    )
    print(f"  Total fine-tuning experiments: {len(finetune_combinations)}")
    print(f"  Results directory: {BASE_RESULTS_DIR}\n")
    
    # Phase 1: Pre-training
    checkpoint_paths, pretrain_results = run_pretraining_experiments(
        PRETRAIN_WD_VALUES,
        samples_per_shape=PRETRAIN_SAMPLES_PER_SHAPE,
        optimization_steps=PRETRAIN_STEPS,
        base_results_dir=BASE_RESULTS_DIR
    )
    
    # Phase 2: Fine-tuning
    finetune_results = run_finetuning_experiments(
        checkpoint_paths,
        finetune_combinations,
        train_points=FINETUNE_TRAIN_POINTS,
        optimization_steps=FINETUNE_STEPS,
        base_results_dir=BASE_RESULTS_DIR
    )
    
    # Save summary
    save_summary(pretrain_results, finetune_results, BASE_RESULTS_DIR)
    
    # Print results table
    print_results_table(finetune_results)
    
    print("\n" + "=" * 100)
    print("EXPERIMENT COMPLETE!")
    print("=" * 100)
    print(f"\nAll results saved to: {BASE_RESULTS_DIR}/")
    print(f"  - Pre-training results: {BASE_RESULTS_DIR}/pretraining/")
    print(f"  - Fine-tuning results: {BASE_RESULTS_DIR}/finetuning/")
    print(f"  - Checkpoints: {BASE_RESULTS_DIR}/checkpoints/")
    print(f"  - Summary: {BASE_RESULTS_DIR}/experiment_summary.json")


if __name__ == "__main__":
    main()
