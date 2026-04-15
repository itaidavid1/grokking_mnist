from argparse import ArgumentParser
import os
import sys

if __package__ in (None, ""):
    # Make package imports work when this file is run directly.
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from grokking_mnist.config import GrokConfig
    from grokking_mnist.train import train
else:
    from .config import GrokConfig
    from .train import train


def parse_args() -> GrokConfig:
    parser = ArgumentParser(description="Grokking experiment on MNIST with optional Grokfast filtering")

    # Training
    parser.add_argument("--label", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimization_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["AdamW", "Adam", "SGD"])
    parser.add_argument("--loss_function", type=str, default="MSE", choices=["CrossEntropy", "MSE"])

    # Model
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--activation", type=str, default="ReLU", choices=["ReLU", "Tanh", "Sigmoid", "GELU"])
    parser.add_argument("--initialization_scale", type=float, default=8.0)

    # Data
    parser.add_argument("--train_points", type=int, default=1000)
    parser.add_argument("--download_directory", type=str, default="./data")

    # Grokfast
    parser.add_argument("--filter", type=str, default="ema", choices=["none", "ma", "ema"])
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--lamb", type=float, default=5.0)

    # Logging
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--log_freq_divisor", type=int, default=150)

    args = parser.parse_args()
    return GrokConfig(**vars(args))


def build_label(config: GrokConfig) -> str:
    """Auto-generate an experiment label from config when none is provided."""
    filter_str = ("_" if config.label != "" else "") + config.filter
    window_size_str = f"_w{config.window_size}"
    alpha_str = f"_a{config.alpha:.3f}".replace(".", "")
    lamb_str = f"_l{config.lamb:.2f}".replace(".", "")

    if config.filter == "none":
        filter_suffix = ""
    elif config.filter == "ma":
        filter_suffix = window_size_str + lamb_str
    elif config.filter == "ema":
        filter_suffix = alpha_str + lamb_str
    else:
        raise ValueError(f"Unrecognized filter type {config.filter}")

    optim_suffix = ""
    if config.weight_decay != 0:
        optim_suffix += f"_wd{config.weight_decay:.1e}".replace(".", "")
    if config.lr != 1e-3:
        optim_suffix += f"_lrx{int(config.lr / 1e-3)}"

    return config.label + filter_str + filter_suffix + optim_suffix


def main():
    config = parse_args()
    config.label = build_label(config)
    print(f"Experiment results saved under name: {config.label}")
    results, model = train(config)


if __name__ == "__main__":
    main()
