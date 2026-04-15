# Comprehensive Pre-training and Fine-tuning Experiment Guide

This guide explains how to run the full experimental pipeline that tests different combinations of weight decay during pre-training and fine-tuning.

## Overview

The experiment investigates how L2 regularization (weight decay) during pre-training on geometric shapes affects subsequent fine-tuning performance on MNIST.

### Experimental Design

**Phase 1: Pre-training on Geometric Shapes**
- 5 different weight decay values: [0.0, 0.01, 0.05, 0.1, 0.2]
- 250 samples per shape (1,000 total: lines, circles, triangles, squares)
- 10,000 optimization steps
- 4 output classes

**Phase 2: Fine-tuning on MNIST**
- For each pre-trained checkpoint:
  - 3 different weight decay fractions: [1/10, 1/100, 1/1000] of pre-training WD
  - 1,000 MNIST training samples
  - 20,000 optimization steps
  - 10 output classes

**Total Experiments:** 5 × 3 = **15 combinations**

## Quick Start

### 1. Test the Pipeline (Recommended First Step)

Run a mini experiment with reduced parameters (4 experiments, ~10-15 minutes):

```bash
python -m grokking_mnist.run_mini_experiment
```

This validates the entire pipeline before committing to the full experiment.

### 2. Run the Full Experiment

Once you're satisfied with the mini test, run the full experiment (~2-4 hours):

```bash
python -m grokking_mnist.run_full_experiment
```

### 3. Analyze Results

After completion, generate visualizations and find the best configurations:

```bash
python -m grokking_mnist.analyze_experiment
```

## Output Structure

Results are organized in `./results/full_experiment/`:

```
full_experiment/
├── pretraining/               # Pre-training results for each weight decay
│   ├── acc_pretrain_wd*.png
│   ├── loss_pretrain_wd*.png
│   └── res_pretrain_wd*.pt
├── finetuning/                # Fine-tuning results for each combination
│   ├── acc_ft_pre*_ft*.png
│   ├── loss_ft_pre*_ft*.png
│   └── res_ft_pre*_ft*.pt
├── checkpoints/               # Saved model checkpoints
│   └── pretrain_wd*.pt
├── experiment_summary.json    # All results in JSON format
├── experiment_heatmaps.png    # Heatmap visualizations
└── experiment_trends.png      # Line plot trends
```

## Understanding the Results

### Experiment Summary JSON

Contains structured data for all experiments:

```json
{
  "timestamp": "2024-...",
  "pretraining": [
    {
      "pretrain_wd": 0.1,
      "final_train_acc": 0.54,
      "final_test_acc": 0.31,
      "checkpoint": "path/to/checkpoint.pt"
    }
  ],
  "finetuning": [
    {
      "pretrain_wd": 0.1,
      "finetune_wd": 0.01,
      "finetune_fraction": 0.1,
      "final_train_acc": 0.92,
      "final_test_acc": 0.85,
      "final_train_loss": 5.2,
      "final_test_loss": 8.1
    }
  ]
}
```

### Visualizations

**Heatmaps** (`experiment_heatmaps.png`):
- Test Accuracy: Higher is better
- Train Accuracy: Compare with test to assess overfitting
- Generalization Gap: Lower is better (less overfitting)

**Trend Plots** (`experiment_trends.png`):
- Shows how performance changes with pre-training weight decay
- Different lines represent different fine-tuning weight decay fractions

## Key Questions the Experiment Addresses

1. **Does pre-training weight decay improve fine-tuning performance?**
   - Compare test accuracy across different pre-training WD values

2. **What's the optimal weight decay for pre-training?**
   - Look at the "Top 5 Configurations" in the analysis output

3. **How should fine-tuning weight decay relate to pre-training weight decay?**
   - Compare fractions (1/10, 1/100, 1/1000) to see which works best

4. **Does pre-training reduce overfitting on MNIST?**
   - Compare generalization gaps for different configurations

## Customization

To run with different parameters, edit `run_full_experiment.py`:

```python
# Configuration section in main()
PRETRAIN_WD_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2]  # Modify as needed
FINETUNE_FRACTIONS = [1/10, 1/100, 1/1000]        # Modify as needed
PRETRAIN_SAMPLES_PER_SHAPE = 250                   # Increase/decrease
PRETRAIN_STEPS = 10000                             # Increase/decrease
FINETUNE_TRAIN_POINTS = 1000                       # Increase/decrease
FINETUNE_STEPS = 20000                             # Increase/decrease
```

## Checkpoint Management

### Saving Checkpoints

Checkpoints are automatically saved after each pre-training run. Each checkpoint contains:
- Model state dict (all layer weights)
- Configuration used for training

### Loading Checkpoints

The fine-tuning phase automatically:
1. Loads the pre-trained checkpoint
2. Transfers weights for matching layers (all except final layer)
3. Initializes the final layer randomly (4 → 10 classes)
4. Fine-tunes on MNIST

### Manual Checkpoint Loading

To manually load and inspect a checkpoint:

```python
import torch
from grokking_mnist.model import build_mlp
from grokking_mnist.config import GrokConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("path/to/checkpoint.pt")

# Create model matching the checkpoint
config = GrokConfig(**checkpoint['config'])
model = build_mlp(config, device, num_classes=4)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
```

## Tips for Running Experiments

1. **Start with the mini experiment** to verify everything works
2. **Monitor the first few pre-training runs** to ensure they're progressing normally
3. **Check disk space** - results can be ~500MB for the full experiment
4. **Use a terminal multiplexer** (like tmux) if running on a remote server
5. **GPU recommended** but not required - CPU will work but take longer

## Troubleshooting

**Out of Memory:**
- Reduce batch size in the config
- Reduce model width/depth
- Use CPU instead of GPU

**Taking too long:**
- Run the mini experiment instead
- Reduce optimization steps
- Reduce samples per shape / MNIST train points

**Results look unusual:**
- Check that shapes are generating correctly: `python -m grokking_mnist.visualize_shapes`
- Verify checkpoint loading: `python -m grokking_mnist.test_checkpoint`
- Check individual result plots in the results directories

## Expected Runtime

**Mini Experiment:**
- 2 pre-training runs: ~2-4 minutes
- 4 fine-tuning runs: ~8-12 minutes
- Total: ~10-15 minutes

**Full Experiment:**
- 5 pre-training runs: ~10-20 minutes
- 15 fine-tuning runs: ~90-180 minutes
- Total: ~2-4 hours (varies by hardware)

## Citation

If you use this experimental framework, please cite the grokking and grokfast papers:

```bibtex
@article{power2022grokking,
  title={Grokking: Generalization beyond overfitting on small algorithmic datasets},
  author={Power, Alethea and Burda, Yuri and Edwards, Harri and Babuschkin, Igor and Misra, Vedant},
  journal={arXiv preprint arXiv:2201.02177},
  year={2022}
}

@article{lee2024grokfast,
  title={Grokfast: Accelerated Grokking by Amplifying Slow Gradients},
  author={Lee, Jaerin and Shin, Bong Gyun and Do, Youngseog and Kim, Jintae and Kim, Keon Lee},
  journal={arXiv preprint arXiv:2405.20233},
  year={2024}
}
```
