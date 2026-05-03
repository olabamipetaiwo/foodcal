"""
model.py

Lightweight MLP classifier for calorie range prediction.

Supports 1 and 2 hidden-layer configurations as described in the paper's
design decisions section.
"""

import torch
import torch.nn as nn
from typing import List


class MLPClassifier(nn.Module):
    """
    Configurable MLP for 3-class calorie range classification.

    Args:
        input_dim:   dimension of the input feature vector
        hidden_dims: list of hidden layer sizes (1 or 2 elements)
        num_classes: number of output classes (default 3: Low/Medium/High)
        dropout:     dropout probability applied after each hidden layer
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256],
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(input_dim: int, num_hidden_layers: int = 2, dropout: float = 0.3) -> MLPClassifier:
    """
    Factory that matches the two MLP depth configurations in the ablation.

    num_hidden_layers=1  →  single 512-unit hidden layer
    num_hidden_layers=2  →  two hidden layers [512, 256]
    """
    if num_hidden_layers == 1:
        hidden_dims = [512]
    elif num_hidden_layers == 2:
        hidden_dims = [512, 256]
    else:
        raise ValueError(f"num_hidden_layers must be 1 or 2, got {num_hidden_layers}")
    return MLPClassifier(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
