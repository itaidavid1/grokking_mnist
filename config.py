from dataclasses import dataclass, field

import torch.nn as nn
import torch.optim as optim


OPTIMIZER_DICT = {
    "AdamW": optim.AdamW,
    "Adam": optim.Adam,
    "SGD": optim.SGD,
}

ACTIVATION_DICT = {
    "ReLU": nn.ReLU,
    "Tanh": nn.Tanh,
    "Sigmoid": nn.Sigmoid,
    "GELU": nn.GELU,
}

LOSS_FUNCTION_DICT = {
    "MSE": nn.MSELoss,
    "CrossEntropy": nn.CrossEntropyLoss,
}


@dataclass
class GrokConfig:
    # Training
    seed: int = 0
    optimization_steps: int = 100_000
    batch_size: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "AdamW"
    loss_function: str = "MSE"

    # Model
    depth: int = 3
    width: int = 200
    activation: str = "ReLU"
    initialization_scale: float = 8.0

    # Data
    train_points: int = 1000
    download_directory: str = "./data"

    # Grokfast
    filter: str = "none"
    alpha: float = 0.99
    window_size: int = 100
    lamb: float = 5.0

    # Logging
    label: str = ""
    results_dir: str = "./results"
    log_freq_divisor: int = 150
