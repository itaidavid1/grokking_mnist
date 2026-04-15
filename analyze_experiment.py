"""Analyze and visualize results from the full experiment.

This script loads the experiment summary and creates visualizations comparing
the 15 different pre-training/fine-tuning combinations.
"""
import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from grokking_mnist.grokking_detector import analyze_grokking_conditions
else:
    from .grokking_detector import analyze_grokking_conditions


def load_results(summary_path="./results/full_experiment/experiment_summary.json"):
    """Load experiment results from JSON file."""
    if not os.path.exists(summary_path):
        print(f"Error: Summary file not found at {summary_path}")
        print("Please run the full experiment first using run_full_experiment.py")
        return None
    
    with open(summary_path, 'r') as f:
        data = json.load(f)
    
    return data


def create_heatmap(results, output_dir):
    """Create heatmap showing test accuracy for all combinations."""
    # Extract unique pretrain and finetune weight decays
    pretrain_wds = sorted(set(r['pretrain_wd'] for r in results))
    finetune_fractions = sorted(set(r['finetune_fraction'] for r in results), reverse=True)
    
    # Create matrix for test accuracies
    test_acc_matrix = np.zeros((len(finetune_fractions), len(pretrain_wds)))
    train_acc_matrix = np.zeros((len(finetune_fractions), len(pretrain_wds)))
    gap_matrix = np.zeros((len(finetune_fractions), len(pretrain_wds)))
    grokking_matrix = np.zeros((len(finetune_fractions), len(pretrain_wds)))
    
    for result in results:
        i = finetune_fractions.index(result['finetune_fraction'])
        j = pretrain_wds.index(result['pretrain_wd'])
        test_acc_matrix[i, j] = result['final_test_acc'] * 100
        train_acc_matrix[i, j] = result['final_train_acc'] * 100
        gap_matrix[i, j] = (result['final_train_acc'] - result['final_test_acc']) * 100
        grokking_matrix[i, j] = 1 if result.get('grokking_info', {}).get('grokking_detected', False) else 0
    
    # Create heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Test Accuracy Heatmap
    im1 = axes[0].imshow(test_acc_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    axes[0].set_title('Test Accuracy (%)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Pre-training Weight Decay', fontsize=12)
    axes[0].set_ylabel('Fine-tuning WD Fraction', fontsize=12)
    axes[0].set_xticks(range(len(pretrain_wds)))
    axes[0].set_xticklabels([f'{wd:.4f}' for wd in pretrain_wds])
    axes[0].set_yticks(range(len(finetune_fractions)))
    axes[0].set_yticklabels([f'1/{int(1/f)}' for f in finetune_fractions])
    
    # Add text annotations
    for i in range(len(finetune_fractions)):
        for j in range(len(pretrain_wds)):
            text = axes[0].text(j, i, f'{test_acc_matrix[i, j]:.1f}',
                              ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im1, ax=axes[0])
    
    # Train Accuracy Heatmap
    im2 = axes[1].imshow(train_acc_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    axes[1].set_title('Train Accuracy (%)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Pre-training Weight Decay', fontsize=12)
    axes[1].set_ylabel('Fine-tuning WD Fraction', fontsize=12)
    axes[1].set_xticks(range(len(pretrain_wds)))
    axes[1].set_xticklabels([f'{wd:.4f}' for wd in pretrain_wds])
    axes[1].set_yticks(range(len(finetune_fractions)))
    axes[1].set_yticklabels([f'1/{int(1/f)}' for f in finetune_fractions])
    
    for i in range(len(finetune_fractions)):
        for j in range(len(pretrain_wds)):
            text = axes[1].text(j, i, f'{train_acc_matrix[i, j]:.1f}',
                              ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im2, ax=axes[1])
    
    # Generalization Gap Heatmap
    im3 = axes[2].imshow(gap_matrix, cmap='RdYlGn_r', aspect='auto')
    axes[2].set_title('Generalization Gap (%)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Pre-training Weight Decay', fontsize=12)
    axes[2].set_ylabel('Fine-tuning WD Fraction', fontsize=12)
    axes[2].set_xticks(range(len(pretrain_wds)))
    axes[2].set_xticklabels([f'{wd:.4f}' for wd in pretrain_wds])
    axes[2].set_yticks(range(len(finetune_fractions)))
    axes[2].set_yticklabels([f'1/{int(1/f)}' for f in finetune_fractions])
    
    for i in range(len(finetune_fractions)):
        for j in range(len(pretrain_wds)):
            text = axes[2].text(j, i, f'{gap_matrix[i, j]:.1f}',
                              ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im3, ax=axes[2])
    
    # Grokking Detection Heatmap
    im4 = axes[3].imshow(grokking_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[3].set_title('Grokking Detected', fontsize=14, fontweight='bold')
    axes[3].set_xlabel('Pre-training Weight Decay', fontsize=12)
    axes[3].set_ylabel('Fine-tuning WD Fraction', fontsize=12)
    axes[3].set_xticks(range(len(pretrain_wds)))
    axes[3].set_xticklabels([f'{wd:.4f}' for wd in pretrain_wds])
    axes[3].set_yticks(range(len(finetune_fractions)))
    axes[3].set_yticklabels([f'1/{int(1/f)}' for f in finetune_fractions])
    
    for i in range(len(finetune_fractions)):
        for j in range(len(pretrain_wds)):
            text = axes[3].text(j, i, 'YES' if grokking_matrix[i, j] == 1 else 'NO',
                              ha="center", va="center", color="black", fontsize=9, fontweight='bold')
    
    plt.colorbar(im4, ax=axes[3])
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'experiment_heatmaps.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Heatmaps saved to: {output_path}")
    plt.close()


def create_line_plots(results, output_dir):
    """Create line plots showing trends."""
    pretrain_wds = sorted(set(r['pretrain_wd'] for r in results))
    finetune_fractions = sorted(set(r['finetune_fraction'] for r in results))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Test accuracy vs pretrain WD for each finetune fraction
    for fraction in finetune_fractions:
        test_accs = []
        for pwd in pretrain_wds:
            result = next(r for r in results 
                         if r['pretrain_wd'] == pwd and r['finetune_fraction'] == fraction)
            test_accs.append(result['final_test_acc'] * 100)
        
        axes[0].plot(pretrain_wds, test_accs, marker='o', 
                    label=f'FT WD = 1/{int(1/fraction)} × Pre WD', linewidth=2)
    
    axes[0].set_xlabel('Pre-training Weight Decay', fontsize=12)
    axes[0].set_ylabel('Test Accuracy (%)', fontsize=12)
    axes[0].set_title('Test Accuracy vs Pre-training Weight Decay', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Generalization gap
    for fraction in finetune_fractions:
        gaps = []
        for pwd in pretrain_wds:
            result = next(r for r in results 
                         if r['pretrain_wd'] == pwd and r['finetune_fraction'] == fraction)
            gap = (result['final_train_acc'] - result['final_test_acc']) * 100
            gaps.append(gap)
        
        axes[1].plot(pretrain_wds, gaps, marker='o', 
                    label=f'FT WD = 1/{int(1/fraction)} × Pre WD', linewidth=2)
    
    axes[1].set_xlabel('Pre-training Weight Decay', fontsize=12)
    axes[1].set_ylabel('Generalization Gap (%)', fontsize=12)
    axes[1].set_title('Generalization Gap vs Pre-training Weight Decay', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'experiment_trends.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Trend plots saved to: {output_path}")
    plt.close()


def find_best_combinations(results, top_n=5):
    """Find the best performing combinations."""
    sorted_results = sorted(results, key=lambda x: x['final_test_acc'], reverse=True)
    
    print("\n" + "=" * 120)
    print(f"TOP {top_n} CONFIGURATIONS BY TEST ACCURACY")
    print("=" * 120)
    print(f"\n{'Rank':<6} {'Pre WD':<10} {'FT WD':<12} {'FT Frac':<12} "
          f"{'Test Acc':<12} {'Train Acc':<12} {'Gap':<10} {'Grokking':<10}")
    print("-" * 120)
    
    for i, result in enumerate(sorted_results[:top_n], 1):
        pretrain_wd = result['pretrain_wd']
        finetune_wd = result['finetune_wd']
        fraction = result['finetune_fraction']
        test_acc = result['final_test_acc'] * 100
        train_acc = result['final_train_acc'] * 100
        gap = train_acc - test_acc
        grokked = "YES" if result.get('grokking_info', {}).get('grokking_detected', False) else "NO"
        
        print(f"{i:<6} {pretrain_wd:<10.4f} {finetune_wd:<12.6f} 1/{int(1/fraction):<10} "
              f"{test_acc:<12.2f} {train_acc:<12.2f} {gap:<10.2f} {grokked:<10}")
    
    print("=" * 120)


def main():
    """Analyze and visualize experiment results."""
    results_dir = "./results/full_experiment"
    summary_path = f"{results_dir}/experiment_summary.json"
    
    print("=" * 100)
    print("EXPERIMENT ANALYSIS")
    print("=" * 100)
    
    # Load results
    data = load_results(summary_path)
    if data is None:
        return
    
    print(f"\nLoaded results from: {summary_path}")
    print(f"Timestamp: {data['timestamp']}")
    print(f"Pre-training experiments: {len(data['pretraining'])}")
    print(f"Fine-tuning experiments: {len(data['finetuning'])}")
    
    # Find best combinations
    find_best_combinations(data['finetuning'], top_n=5)
    
    # Analyze grokking conditions
    analyze_grokking_conditions(data['finetuning'], verbose=True)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_heatmap(data['finetuning'], results_dir)
    create_line_plots(data['finetuning'], results_dir)
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE!")
    print("=" * 100)
    print(f"\nVisualizations saved to: {results_dir}/")
    print(f"  - experiment_heatmaps.png")
    print(f"  - experiment_trends.png")


if __name__ == "__main__":
    main()
