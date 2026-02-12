import torch
import torch.nn as nn
import torch.nn.functional as F


class Graph_ReLu_W_WithPrior(nn.Module):

    def __init__(
        self,
        n_nodes,
        k,
        device,
        A_prior=None,
    ):
        super().__init__()

        self.k = k
        self.device = device

        if A_prior is not None:
            # initialize W with ground truth
            self.A_param = nn.Parameter(
                A_prior.clone().to(device)
            )
        else:
            self.A_param = nn.Parameter(
                torch.randn(n_nodes, n_nodes, device=device)
            )

    def forward(self, idx):

        adj = F.relu(self.A_param)

        if self.k:
            mask = torch.zeros_like(adj)
            _, indices = adj.topk(self.k, dim=1)
            mask.scatter_(1, indices, 1.0)
            adj = adj * mask

        return adj

