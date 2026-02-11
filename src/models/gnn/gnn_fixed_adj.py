import torch
import torch.nn as nn


class GCLayer(nn.Module):
    """
    Graph convolution layer (same as paper code).
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)

    def forward(self, adj, X):
        # add self-loops
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        h = self.dense(X)  # [B, N, out_dim] if X is [B, N, in_dim]
        norm = adj.sum(1).clamp(min=1e-12).pow(-0.5)
        # normalized adjacency multiplication
        h = norm[None, :] * adj * norm[:, None] @ h
        return h


class GNN_TAM_FixedAdj(nn.Module):
    """
    Same architecture as GNN_TAM, but adjacency is fixed (no GSL).
    X expected: [B, N, T] where T=window_size, N=n_nodes
    """
    def __init__(
        self,
        n_nodes: int,
        window_size: int,
        n_classes: int,
        adj_fixed,
        n_gnn: int = 1,
        n_hidden: int = 1024,
        device: str = "cpu",
    ):
        super().__init__()
        self.window_size = window_size
        self.nhidden = n_hidden
        self.device = device
        self.n_gnn = n_gnn

        # buffer adjacency (fixed, not trainable)
        if isinstance(adj_fixed, torch.Tensor):
            A = adj_fixed.float().to(device)
        else:
            A = torch.tensor(adj_fixed, dtype=torch.float32, device=device)

        if A.shape != (n_nodes, n_nodes):
            raise ValueError(f"adj_fixed shape {tuple(A.shape)} != ({n_nodes}, {n_nodes})")

        # same z-mask as paper: remove diagonal contributions from learned/used adjacency
        self.z = (torch.ones(n_nodes, n_nodes, device=device) - torch.eye(n_nodes, device=device))
        A = A * self.z

        # register as buffer so it moves with .to() and is saved in state_dict
        self.register_buffer("A_fixed", A)

        # lists same as paper
        self.adj = [0 for _ in range(n_gnn)]
        self.h = [0 for _ in range(n_gnn)]
        self.skip = [0 for _ in range(n_gnn)]

        self.conv1 = nn.ModuleList()
        self.bnorm1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.bnorm2 = nn.ModuleList()

        for _ in range(self.n_gnn):
            self.conv1.append(GCLayer(window_size, n_hidden))
            self.bnorm1.append(nn.BatchNorm1d(n_nodes))
            self.conv2.append(GCLayer(n_hidden, n_hidden))
            self.bnorm2.append(nn.BatchNorm1d(n_nodes))

        self.fc = nn.Linear(n_gnn * n_hidden, n_classes)

    def forward(self, X):
        """
        X: [B, N, T]
        """
        X = X.to(self.device)
        for i in range(self.n_gnn):
            self.adj[i] = self.A_fixed  # fixed adjacency
            self.h[i] = self.conv1[i](self.adj[i], X).relu()
            self.h[i] = self.bnorm1[i](self.h[i])

            self.skip[i], _ = torch.min(self.h[i], dim=1)

            self.h[i] = self.conv2[i](self.adj[i], self.h[i]).relu()
            self.h[i] = self.bnorm2[i](self.h[i])

            self.h[i], _ = torch.min(self.h[i], dim=1)
            self.h[i] = self.h[i] + self.skip[i]

        h = torch.cat(self.h, dim=1)
        output = self.fc(h)
        return output

    def get_adj(self):
        return self.adj
