"""Mini experiment for quick testing (2 pretrain WD × 2 finetune WD = 4 experiments).

This is a scaled-down version of run_full_experiment.py for quick testing:
- Pre-train: 2 weight decays [0.0, 0.1]
- Fine-tune: 2 weight decay fractions [1/10, 1/100]
- Fewer samples and steps
- Total: 4 experiments instead of 15

Use this to test the pipeline before running the full experiment.
"""
import os
import sys
import json

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from grokking_mnist.run_full_experiment import (
        run_pretraining_experiments,
        run_finetuning_experiments,
        save_summary,
        print_results_table
    )
    from grokking_mnist.grokking_detector import analyze_grokking_conditions
else:
    from .run_full_experiment import (
        run_pretraining_experiments,
        run_finetuning_experiments,
        save_summary,
        print_results_table
    )
    from .grokking_detector import analyze_grokking_conditions


def main():
    """Run the mini experiment."""
    # Configuration (reduced for quick testing)
    PRETRAIN_WD_VALUES = [0.0, 0.1]  # 2 different weight decays
    FINETUNE_FRACTIONS = [1/10, 1/100]  # 2 fractions
    
    BASE_RESULTS_DIR = "./results/mini_experiment"
    PRETRAIN_SAMPLES_PER_SHAPE = 100  # Reduced from 250
    PRETRAIN_STEPS = 2000  # Reduced from 10000
    FINETUNE_TRAIN_POINTS = 500  # Reduced from 1000
    FINETUNE_STEPS = 3000  # Reduced from 20000
    
    print("=" * 100)
    print("MINI PRE-TRAINING AND FINE-TUNING EXPERIMENT (FOR TESTING)")
    print("=" * 100)
    print(f"\nConfiguration:")
    print(f"  Pre-training weight decays: {PRETRAIN_WD_VALUES}")
    print(f"  Fine-tuning fractions: {[f'1/{int(1/f)}' for f in FINETUNE_FRACTIONS]}")
    print(f"  Total experiments: {len(PRETRAIN_WD_VALUES)} × {len(FINETUNE_FRACTIONS)} = {len(PRETRAIN_WD_VALUES) * len(FINETUNE_FRACTIONS)}")
    print(f"  Results directory: {BASE_RESULTS_DIR}")
    print(f"\n  NOTE: This is a reduced version for testing.")
    print(f"  For full experiments, run: python -m grokking_mnist.run_full_experiment\n")
    
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
        FINETUNE_FRACTIONS,
        PRETRAIN_WD_VALUES[0],
        train_points=FINETUNE_TRAIN_POINTS,
        optimization_steps=FINETUNE_STEPS,
        base_results_dir=BASE_RESULTS_DIR
    )
    
    # Save summary
    save_summary(pretrain_results, finetune_results, BASE_RESULTS_DIR)
    
    # Print results table
    print_results_table(finetune_results)
    
    # Analyze grokking
    analyze_grokking_conditions(finetune_results, verbose=True)
    
    print("\n" + "=" * 100)
    print("MINI EXPERIMENT COMPLETE!")
    print("=" * 100)
    print(f"\nAll results saved to: {BASE_RESULTS_DIR}/")
    print(f"\nTo run the full experiment with 15 total experiments:")
    print(f"  python -m grokking_mnist.run_full_experiment")


if __name__ == "__main__":
    main()
