# Complete Comparison Setup

This document describes the three-way comparison setup to evaluate:
1. Effect of pre-training on geometric shapes
2. Effect of Grokfast gradient filtering
3. Baseline performance

## Three Experiments (All Fair Comparison)

### Common Settings Across All Experiments:
- **MNIST Training Data**: 1000 samples
- **Optimization Steps on MNIST**: 20,000
- **Model Architecture**: 3-layer MLP, width 200, ReLU activation
- **Optimizer**: AdamW with lr=1e-3 (or 5e-4 for fine-tuning)
- **Loss Function**: MSE
- **Batch Size**: 50
- **Seed**: 42 (reproducible)

### Experiment 1: Pre-training + Fine-tuning (NO Grokfast)

**Purpose**: Test if pre-training on geometric shapes helps MNIST performance

**Script**: `run_full_experiment.py`

**Setup**:
- Pre-train on 1000 geometric shapes (250 per shape type)
- Test 5 pre-training weight decays: [0.0, 0.01, 0.05, 0.1, 1.0]
- Fine-tune on MNIST with 3 weight decay fractions: [1/10, 1/100, 1/1000]
- Total: 15 experiments (5 × 3)
- **Grokfast**: DISABLED (both pre-training and fine-tuning)
- Results: `./results/pretrain_finetune/`

**Run**:
```bash
python -m grokking_mnist.run_full_experiment
```

**Duration**: ~30-60 minutes (15 experiments)

---

### Experiment 2: Grokfast (Direct MNIST)

**Purpose**: Test Grokfast gradient filtering effect on direct MNIST training

**Script**: `run_baseline_comparison.py` (grokfast part)

**Setup**:
- Direct training on MNIST (no pre-training)
- 1000 MNIST samples, 20,000 steps
- **Grokfast**: ENABLED (EMA filter, alpha=0.99, lamb=5.0)
- Weight decay: 0.0
- Results: `./results/baseline_vs_grokfast/grokfast/`

**Run**:
```bash
python -m grokking_mnist.run_baseline_comparison
```
(This runs both baseline and grokfast together)

**Duration**: ~5-10 minutes (2 experiments)

---

### Experiment 3: Baseline (Direct MNIST)

**Purpose**: Pure baseline performance without any special techniques

**Script**: `run_baseline_comparison.py` (baseline part)

**Setup**:
- Direct training on MNIST (no pre-training)
- 1000 MNIST samples, 20,000 steps
- **Grokfast**: DISABLED
- Weight decay: 0.0
- Results: `./results/baseline_vs_grokfast/baseline/`

**Run**: Same as Experiment 2 (runs both together)

---

## Running the Full Comparison

### Step 1: Run Pre-training Experiments
```bash
python -m grokking_mnist.run_full_experiment
```
This gives you 15 pre-training + fine-tuning results.

### Step 2: Run Baseline vs Grokfast
```bash
python -m grokking_mnist.run_baseline_comparison
```
This gives you 2 direct MNIST training results (with/without Grokfast).

### Step 3: Analyze All Results

For pre-training experiments:
```bash
python -m grokking_mnist.analyze_experiment
```

The baseline vs grokfast comparison automatically prints a summary table.

---

## Expected Results Directory Structure

```
results/
├── pretrain_finetune/              # Experiment 1
│   ├── pretraining/
│   │   ├── acc_*.png
│   │   ├── loss_*.png
│   │   └── res_*.pt
│   ├── finetuning/
│   │   ├── acc_*.png
│   │   ├── loss_*.png
│   │   └── res_*.pt
│   ├── checkpoints/
│   │   └── pretrain_wd*.pt
│   └── experiment_summary.json
│
└── baseline_vs_grokfast/           # Experiments 2 & 3
    ├── baseline/
    │   ├── acc_baseline.png
    │   ├── loss_baseline.png
    │   └── res_baseline.pt
    ├── grokfast/
    │   ├── acc_grokfast.png
    │   ├── loss_grokfast.png
    │   └── res_grokfast.pt
    └── comparison_summary.json
```

---

## What Each Experiment Tests

### Comparison A: Pre-training Effect
**Compare**: Experiment 1 (best config) vs Experiment 3 (baseline)

**Question**: Does pre-training on geometric shapes improve MNIST performance?

**Expected**: If pre-training helps, Experiment 1 should show:
- Higher test accuracy than baseline
- Better generalization (smaller train-test gap)
- Potentially different grokking behavior

### Comparison B: Grokfast Effect
**Compare**: Experiment 2 (grokfast) vs Experiment 3 (baseline)

**Question**: Does Grokfast accelerate/improve grokking?

**Expected**: If Grokfast works, Experiment 2 should show:
- Earlier grokking transition
- Potentially higher final test accuracy
- More pronounced grokking pattern

### Comparison C: Pre-training + Grokfast (Optional)
If you want to test pre-training WITH Grokfast, you can modify `run_full_experiment.py` to enable Grokfast again and save to a different directory.

---

## Key Metrics to Compare

For each experiment, look at:
1. **Final Test Accuracy**: Higher is better
2. **Generalization Gap**: train_acc - test_acc (lower is better)
3. **Grokking Detection**: Did grokking occur?
4. **Grokking Step**: When did grokking happen? (earlier is better)
5. **Test Accuracy Improvement**: Change from early to late training

---

## Quick Comparison Commands

Check if grokking occurred in any result:
```bash
# Check baseline
python -m grokking_mnist.check_grokking ./results/baseline_vs_grokfast/baseline/res_baseline.pt

# Check grokfast
python -m grokking_mnist.check_grokking ./results/baseline_vs_grokfast/grokfast/res_grokfast.pt

# Check a pre-training fine-tuning result
python -m grokking_mnist.check_grokking ./results/pretrain_finetune/finetuning/res_*.pt
```

Visualize any result:
```bash
python -m grokking_mnist.visualize_grokking ./results/path/to/res_file.pt
```

---

## Summary Table Format

After running all experiments, you'll see tables like:

```
Pre-training + Fine-tuning Results:
Pretrain WD  Finetune WD  Train Acc  Test Acc  Gap    Grokking
0.0500       0.005000     98.20      93.10     5.10   YES
...

Baseline vs Grokfast Results:
Experiment       Filter     Train Acc    Test Acc     Gap       Grokking
Baseline         none       95.30        88.20        7.10      NO
Grokfast         ema        96.10        91.50        4.60      YES

Grokfast Improvement:
  Test Accuracy Gain: +3.30%
```

This setup allows you to:
1. Isolate the effect of pre-training
2. Isolate the effect of Grokfast
3. Compare all approaches fairly with identical MNIST training conditions
