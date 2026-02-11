import torch
import torch.nn as nn
import torch.nn.functional as F


class Graph_ReLu_W_WithPrior(nn.Module):
    """
    Trainable adjacency with optional prior (ground truth).
    Supports:
        - pure trainable
        - fixed prior
        - residual learning on prior
    """

    def __init__(
        self,
        n_nodes,
        k,
        device,
        A_prior=None,
        lambda_prior=0.1,
        train_residual=False,
    ):
        super().__init__()

        self.k = k
        self.device = device
        self.lambda_prior = lambda_prior
        self.train_residual = train_residual

        # trainable parameter
        self.A_param = nn.Parameter(
            torch.randn(n_nodes, n_nodes, device=device)
        )

        if A_prior is not None:
            self.register_buffer("A_prior", A_prior)
        else:
            self.A_prior = None

    def forward(self, idx):
        A_learned = F.relu(self.A_param)

        if self.A_prior is not None:
            if self.train_residual:
                adj = self.A_prior + self.lambda_prior * A_learned
            else:
                # فقط از prior استفاده کن
                adj = self.A_prior
        else:
            adj = A_learned

        # optional top-k sparsification
        if self.k:
            mask = torch.zeros_like(adj)
            _, indices = adj.topk(self.k, dim=1)
            mask.scatter_(1, indices, 1.0)
            adj = adj * mask

        return adj
