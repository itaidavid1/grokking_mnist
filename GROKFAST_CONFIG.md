# Grokfast Configuration Applied to Full Experiment

## What Changed

The full experiment (`run_full_experiment.py`) now **enables Grokfast gradient filtering** to accelerate and induce grokking behavior during training.

## Grokfast Settings

### Algorithm: EMA (Exponential Moving Average)

Both pre-training and fine-tuning now use:

```python
filter="ema",    # EMA gradient filtering
alpha=0.99,      # Smoothing parameter (higher = more smoothing)
lamb=5.0         # Amplification parameter (higher = stronger effect)
```

### What These Parameters Mean

- **`filter="ema"`**: Uses exponential moving average gradient filtering
  - Maintains a slow-moving average of gradients
  - Amplifies slow-changing gradient components
  - This accelerates the grokking transition

- **`alpha=0.99`**: Smoothing factor for EMA
  - Controls how much history to retain (0.99 = 99% history, 1% new)
  - Higher values = smoother gradient estimates
  - Standard value from Grokfast paper

- **`lamb=5.0`**: Amplification strength
  - Controls how much to boost slow gradients
  - Higher values = stronger grokking acceleration
  - Standard value from Grokfast paper

## Where Applied

### 1. Pre-training on Geometric Shapes

```python
config = GrokConfig(
    ...
    filter="ema",
    alpha=0.99,
    lamb=5.0,
    ...
)
```

Applied to all 5 pre-training runs with different weight decays [0.0, 0.01, 0.05, 0.1, 0.2].

### 2. Fine-tuning on MNIST

```python
config = GrokConfig(
    ...
    filter="ema",
    alpha=0.99,
    lamb=5.0,
    ...
)
```

Applied to all 15 fine-tuning experiments (5 pre-train WD × 3 fine-tune fractions).

## Results Directory

Changed to distinguish from non-Grokfast experiments:

```python
BASE_RESULTS_DIR = "./results/full_experiment_grokfast"
```

This keeps your Grokfast results separate from any previous runs.

## How Grokfast Works

Grokfast accelerates grokking by:

1. **Tracking gradient history**: Maintains exponential moving average of gradients
2. **Identifying slow components**: Finds gradient components that change slowly
3. **Amplifying slow gradients**: Adds amplified slow components back to gradients
4. **Faster generalization**: This speeds up the transition from overfitting to generalization

The intuition is that slow-changing gradients correspond to learning generalizable features, while fast-changing gradients correspond to memorization. By amplifying the slow components, we accelerate the grokking process.

## Expected Behavior

With Grokfast enabled, you should observe:

1. **Earlier grokking transitions**: Models reach good test accuracy faster
2. **More pronounced grokking**: Clearer separation between overfitting and generalization phases
3. **Higher grokking detection rate**: More experiments should exhibit detectable grokking patterns

## Running the Experiment

Simply run as before:

```bash
# Full experiment (15 experiments total)
python -m grokking_mnist.run_full_experiment

# Mini experiment (4 experiments for quick testing)
python -m grokking_mnist.run_mini_experiment

# Analyze results
python -m grokking_mnist.analyze_experiment
```

The Grokfast configuration is now automatically applied!

## Comparing with Non-Grokfast

If you want to compare Grokfast vs non-Grokfast, you can:

1. Run with Grokfast (current setup) → saves to `./results/full_experiment_grokfast/`
2. Temporarily change `filter="ema"` back to `filter="none"` 
3. Change `BASE_RESULTS_DIR` back to `"./results/full_experiment"`
4. Run again to get baseline results
5. Compare the two result directories

## Alternative: Moving Average (MA) Filter

If you want to try the MA filter instead of EMA:

```python
filter="ma",         # Moving average filter
window_size=100,     # Number of gradients to average
lamb=5.0            # Amplification parameter
```

The EMA filter is generally recommended as it's more memory-efficient and performs similarly.

## References

Lee et al. (2024). "Grokfast: Accelerated Grokking by Amplifying Slow Gradients"
- arXiv: https://arxiv.org/abs/2405.20233
- Shows that amplifying slow gradient components accelerates grokking by up to 2-3x
