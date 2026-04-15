import torch
import torch.nn as nn

from .config import ACTIVATION_DICT, GrokConfig


def build_mlp(config: GrokConfig, device: torch.device, num_classes: int = 10) -> nn.Sequential:
    """Build a configurable MLP for image classification (784 -> ... -> num_classes).

    The large ``initialization_scale`` pushes the network into a memorisation
    regime first, which is a prerequisite for observing grokking.
    """
    assert config.activation in ACTIVATION_DICT, (
        f"Unsupported activation: {config.activation}"
    )
    activation_fn = ACTIVATION_DICT[config.activation]

    layers: list[nn.Module] = [nn.Flatten()]
    for i in range(config.depth):
        if i == 0:
            layers.append(nn.Linear(784, config.width))
            layers.append(activation_fn())
        elif i == config.depth - 1:
            layers.append(nn.Linear(config.width, num_classes))
        else:
            layers.append(nn.Linear(config.width, config.width))
            layers.append(activation_fn())

    mlp = nn.Sequential(*layers).to(device)

    with torch.no_grad():
        for p in mlp.parameters():
            p.data *= config.initialization_scale

    return mlp
