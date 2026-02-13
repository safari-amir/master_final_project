import torch
import torch.nn as nn
import torch.nn.functional as F


class Graph_ReLu_W_PartialFreeze(nn.Module):
    """
    Trainable adjacency with partially frozen prior.

    - Edges where mask == 1 → fixed to A_prior
    - Edges where mask == 0 → learned via ReLU(W)
    """

    def __init__(
        self,
        n_nodes,
        device,
        A_prior,
        freeze_mask,   # same shape as adjacency
        k=None,
    ):
        super().__init__()

        self.device = device
        self.k = k

        # learnable parameter
        self.W = nn.Parameter(
            torch.randn(n_nodes, n_nodes, device=device)
        )

        # register buffers (no gradient)
        self.register_buffer("A_prior", A_prior.float())
        self.register_buffer("freeze_mask", freeze_mask.float())

    def forward(self, idx):

        A_learned = F.relu(self.W)

        # freeze logic:
        # where mask=1 → use prior
        # where mask=0 → use learned
        adj = (
            self.A_prior * self.freeze_mask +
            A_learned * (1 - self.freeze_mask)
        )

        # optional top-k
        if self.k:
            mask = torch.zeros_like(adj)
            _, indices = adj.topk(self.k, dim=1)
            mask.scatter_(1, indices, 1.0)
            adj = adj * mask

        return adj
