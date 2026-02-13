import torch
import torch.nn as nn
from .gsl_with_partial_freeze import Graph_ReLu_W_PartialFreeze


class GCLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)

    def forward(self, adj, X):
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        h = self.dense(X)

        deg = adj.sum(1).clamp(min=1e-12)
        norm = deg.pow(-0.5)

        h = norm[None, :] * adj * norm[:, None] @ h
        return h


class GNN_TAM_PartialFreeze(nn.Module):
    """
    Trainable GNN with partially frozen adjacency.

    - Edges defined by freeze_mask are fixed to A_prior
    - Other edges are learned
    """

    def __init__(
        self,
        n_nodes,
        window_size,
        n_classes,
        n_gnn=1,
        n_hidden=1024,
        k=None,
        device="cpu",
        A_prior=None,
        freeze_mask=None,
    ):
        super().__init__()

        self.device = device
        self.idx = torch.arange(n_nodes).to(device)
        self.z = (torch.ones(n_nodes, n_nodes) - torch.eye(n_nodes)).to(device)
        self.n_gnn = n_gnn

        if A_prior is None or freeze_mask is None:
            raise ValueError("A_prior and freeze_mask must be provided")

        self.gsl = nn.ModuleList()
        self.conv1 = nn.ModuleList()
        self.bnorm1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.bnorm2 = nn.ModuleList()

        for _ in range(n_gnn):
            self.gsl.append(
                Graph_ReLu_W_PartialFreeze(
                    n_nodes=n_nodes,
                    device=device,
                    A_prior=A_prior,
                    freeze_mask=freeze_mask,
                    k=k,
                )
            )

            self.conv1.append(GCLayer(window_size, n_hidden))
            self.bnorm1.append(nn.BatchNorm1d(n_nodes))
            self.conv2.append(GCLayer(n_hidden, n_hidden))
            self.bnorm2.append(nn.BatchNorm1d(n_nodes))

        self.fc = nn.Linear(n_gnn * n_hidden, n_classes)

    def forward(self, X):
        X = X.to(self.device)
        h_list = []

        for i in range(self.n_gnn):

            adj = self.gsl[i](self.idx)
            adj = adj * self.z

            # First GCN
            h = self.conv1[i](adj, X).relu()
            h = self.bnorm1[i](h)

            skip, _ = torch.min(h, dim=1)

            # Second GCN
            h = self.conv2[i](adj, h).relu()
            h = self.bnorm2[i](h)
            h, _ = torch.min(h, dim=1)

            h = h + skip
            h_list.append(h)

        h = torch.cat(h_list, dim=1)
        return self.fc(h)

    def get_adj(self):
        return [gsl_layer(self.idx) for gsl_layer in self.gsl]
