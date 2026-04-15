# Grokking MNIST with Geometric Shapes Pre-training

This project implements grokking experiments on MNIST with optional pre-training on geometric shapes.

## Features

- **MNIST Training**: Train models that exhibit grokking behavior on MNIST
- **Grokking Detection**: Automatic detection and analysis of grokking behavior during training
  - Detects early overfitting phase (high train accuracy, low test accuracy)
  - Identifies delayed generalization phase (test accuracy improvement)
  - Pinpoints the grokking transition step
  - Comprehensive reports with metrics and visualizations
- **Geometric Shapes Pre-training**: Pre-train on synthetic shapes with rich variations:
  - **Lines**: Various angles, thicknesses, positions, and sharpness levels
  - **Circles**: Various radii, positions, and sharpness levels
  - **Triangles**: Various sizes, orientations, positions, and sharpness levels
  - **Squares**: Various sizes, orientations, positions, and sharpness levels
  - All shapes appear at **random locations** within the image
  - **High L2 regularization** (weight decay) during pre-training to learn robust features
- **Grokfast Support**: Optional gradient filtering for faster grokking

## Installation

```bash
pip install torch torchvision matplotlib numpy scipy pillow tqdm
```

## Usage

### Pre-training on Geometric Shapes

Run pre-training on geometric shapes with variations:

```bash
python -m grokking_mnist.pretrain_shapes --samples_per_shape 250 --optimization_steps 10000
```

**Pre-training uses high L2 regularization (weight decay = 0.1) by default** to encourage the model to learn robust, generalizable features.

Key arguments:
- `--samples_per_shape`: Number of samples per shape class (default: 250, total=1000)
- `--optimization_steps`: Number of training steps (default: 10000)
- `--batch_size`: Batch size (default: 50)
- `--lr`: Learning rate (default: 1e-3)
- `--weight_decay`: L2 regularization strength (default: 0.1 for pre-training)
- `--filter`: Grokfast filter type: `none`, `ma`, `ema` (default: none)
- `--label`: Custom label for results (default: shapes_pretrain)

Example with custom weight decay:
```bash
python -m grokking_mnist.pretrain_shapes --weight_decay 0.05
```

Example with Grokfast EMA filter:
```bash
python -m grokking_mnist.pretrain_shapes --filter ema --alpha 0.99 --lamb 5.0
```

**Compare different weight decay values**:
```bash
python -m grokking_mnist.compare_weight_decay
```

This runs experiments with various weight decay values (0.0, 0.01, 0.05, 0.1, 0.2) to demonstrate the effect of L2 regularization on generalization.

### Training on MNIST

```bash
python -m grokking_mnist.run --train_points 1000 --optimization_steps 100000
```

### Visualizing Geometric Shapes

See examples of the generated shapes:

```bash
python -m grokking_mnist.visualize_shapes
```

This creates a grid showing 8 samples of each shape type with different variations.

**Test Position Variation**:

```bash
python -m grokking_mnist.test_positions
```

This generates 10 samples of each shape to verify they appear at different locations in the image.

### Complete Comparison Experiments

This project supports three types of experiments for fair comparison:

#### 1. Pre-training + Fine-tuning (Tests Pre-training Effect)

Run 15 experiments testing 5 pre-training weight decays × 3 fine-tuning weight decays:

```bash
python -m grokking_mnist.run_full_experiment
```

This script will:
1. **Pre-train** on geometric shapes with 5 different weight decays: [0.0, 0.01, 0.05, 0.1, 1.0]
2. **Save checkpoints** from each pre-training run
3. **Fine-tune** each checkpoint on MNIST with 3 weight decay fractions: [1/10, 1/100, 1/1000]
4. Save all results with grokking detection
5. **NO Grokfast** (tests pure pre-training effect)

#### 2. Baseline vs Grokfast (Tests Grokfast Effect)

Run 2 experiments comparing direct MNIST training with/without Grokfast:

```bash
python -m grokking_mnist.run_baseline_comparison
```

This script will:
1. **Baseline**: Direct MNIST training WITHOUT Grokfast
2. **Grokfast**: Direct MNIST training WITH Grokfast (EMA filter)
3. Print comparison table showing the difference
4. Save results to `./results/baseline_vs_grokfast/`

Both use identical settings (1000 MNIST samples, 20,000 steps) for fair comparison.

**Analyze the results**:

```bash
python -m grokking_mnist.analyze_experiment
```

This creates visualizations showing:
- Heatmaps of test accuracy, train accuracy, generalization gap, and grokking detection
- Line plots showing trends across different weight decay combinations
- Top 5 best performing configurations
- Summary of which experiments exhibited grokking behavior

Results are saved to `./results/full_experiment/`

**Configuration (All experiments use comparable settings):**
- **Common Settings**: 1,000 MNIST samples, 20,000 training steps on MNIST
- **Pre-training Experiment**:
  - Pre-training weight decays: [0.0, 0.01, 0.05, 0.1, 1.0]
  - Fine-tuning weight decay fractions: [1/10, 1/100, 1/1000]
  - Grokfast: DISABLED
  - Results: `./results/pretrain_finetune/`
- **Baseline vs Grokfast Experiments**:
  - Direct MNIST training (no pre-training)
  - Grokfast: one with, one without
  - Results: `./results/baseline_vs_grokfast/`

See `COMPARISON_SETUP.md` for detailed comparison methodology.

**Quick Test (Mini Experiment)**:

Before running the full experiment (which can take a while), test the pipeline with a mini version:

```bash
python -m grokking_mnist.run_mini_experiment
```

This runs only 4 experiments (2×2) with reduced samples and steps for quick validation

### Grokking Detection and Visualization

**Check if grokking occurred in saved results**:

```bash
python -m grokking_mnist.check_grokking ./results/path/to/res_label.pt
```

This analyzes a saved results file and prints a detailed grokking detection report.

**Visualize training curves with grokking annotations**:

```bash
python -m grokking_mnist.visualize_grokking ./results/path/to/res_label.pt
```

This creates an enhanced visualization showing:
- Training and test accuracy curves
- Grokking transition point (if detected)
- Overfitting and generalization phases (highlighted)
- Key metrics and statistics
- Loss curves

The grokking detector identifies three key patterns:
1. **Early Overfitting**: Train accuracy reaches high values (>90%) while test lags significantly (>15% gap)
2. **Delayed Generalization**: Test accuracy improves substantially later (>10% improvement)
3. **Grokking Transition**: The step where the most rapid test accuracy improvement occurs

## Dataset Variations

The geometric shapes dataset includes the following variations:

### Lines (Label 0)
- **Thickness**: 1-3 pixels
- **Orientation**: 0-180 degrees
- **Position**: Random locations within image bounds
- **Sharpness**: 4 levels (sharp to blurred)

### Circles (Label 1)
- **Radius**: 4-10 pixels
- **Position**: Random locations within image bounds
- **Sharpness**: 4 levels (sharp to blurred)

### Triangles (Label 2)
- **Scale**: 0.6-1.2x base size
- **Orientation**: 0-360 degrees (full rotation)
- **Position**: Random locations within image bounds
- **Sharpness**: 4 levels (sharp to blurred)

### Squares (Label 3)
- **Side Length**: 8-16 pixels
- **Orientation**: 0-90 degrees
- **Position**: Random locations within image bounds
- **Sharpness**: 4 levels (sharp to blurred)

## Output

Results are saved to `./results/` including:
- Training/test accuracy plots
- Training/test loss plots
- Saved metrics (`.pt` files)

## L2 Regularization in Pre-training

Pre-training uses **high weight decay (L2 regularization, default=0.1)** to:
- Prevent overfitting on the synthetic shapes dataset
- Encourage the model to learn simple, robust feature representations
- Promote weight values that generalize well to downstream tasks (like MNIST)

The regularization term adds a penalty proportional to the squared magnitude of weights:

```
Loss = Task_Loss + (weight_decay / 2) * ||weights||²
```

You can adjust the weight decay strength with `--weight_decay`:
- Higher values (e.g., 0.1-0.5): Stronger regularization, simpler features
- Lower values (e.g., 0.01-0.05): Weaker regularization, more complex features
- Zero (0.0): No regularization (not recommended for pre-training)

## Configuration

All training parameters can be configured via command-line arguments or by modifying `config.py`.

## Architecture

- **Model**: Configurable MLP (default: 3 layers, 200 width)
- **Input**: 28x28 grayscale images (784 features)
- **Output**: 4 classes (shapes) or 10 classes (MNIST)
- **Loss**: MSE or CrossEntropy
- **Optimizer**: AdamW, Adam, or SGD
