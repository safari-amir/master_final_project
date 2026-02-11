import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """
    MLP baseline for TEP fault diagnosis.
    Input: [batch, window_size, n_nodes]
    """

    def __init__(
        self,
        window_size: int,
        n_nodes: int,
        n_classes: int,
        hidden_dims=(512, 256),
        dropout: float = 0.2,
    ):
        super().__init__()

        input_dim = window_size * n_nodes
        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, n_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: [batch, window_size, n_nodes]
        """
        x = x.view(x.size(0), -1)  # flatten
        return self.net(x)
