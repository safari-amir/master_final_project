import torch
import torch.nn as nn
import torch.nn.functional as F


class Graph_ReLu_W(nn.Module):
    def __init__(self, n_nodes, k, device):
        super().__init__()
        self.k = k
        self.device = device
        self.A = nn.Parameter(torch.randn(n_nodes, n_nodes, device=device))

    def forward(self, idx):
        adj = F.relu(self.A)

        if self.k:
            mask = torch.zeros_like(adj)
            _, indices = adj.topk(self.k, dim=1)
            mask.scatter_(1, indices, 1.0)
            adj = adj * mask

        return adj


class GSL(nn.Module):
    def __init__(self, gsl_type, n_nodes, window_size, alpha, k, device):
        super().__init__()

        if gsl_type == "relu":
            self.gsl_layer = Graph_ReLu_W(n_nodes, k, device)
        else:
            raise NotImplementedError("Only relu GSL implemented for now.")

    def forward(self, idx):
        return self.gsl_layer(idx)
