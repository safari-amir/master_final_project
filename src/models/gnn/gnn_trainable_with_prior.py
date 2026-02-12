import torch
import torch.nn as nn
from .gsl_with_prior import Graph_ReLu_W_WithPrior


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


class GNN_TAM_TrainableWithPrior(nn.Module):
    """
    Trainable GNN with optional ground-truth initialization
    and residual learning.
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
        lambda_prior=0.1,
        train_residual=False,
    ):
        super().__init__()

        self.device = device
        self.idx = torch.arange(n_nodes).to(device)
        self.z = (torch.ones(n_nodes, n_nodes) - torch.eye(n_nodes)).to(device)
        self.n_gnn = n_gnn

        self.gsl = nn.ModuleList()
        self.conv1 = nn.ModuleList()
        self.bnorm1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.bnorm2 = nn.ModuleList()

        for _ in range(n_gnn):
            self.gsl.append(
                Graph_ReLu_W_WithPrior(
                    n_nodes=n_nodes,
                    k=k,
                    device=device,
                    A_prior=A_prior,
                    #lambda_prior=lambda_prior,
                    #train_residual=train_residual,
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
