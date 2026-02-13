import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from tqdm.auto import trange

from fddbenchmark import FDDDataset, FDDDataloader

# ---- models ----
from src.models.baselines.mlp import MLPBaseline
from src.models.gnn.gnn_fixed_adj import GNN_TAM_FixedAdj
from src.models.gnn.gnn_trainable_adj import GNN_TAM_Trainable
from src.models.gnn.gnn_trainable_with_prior import GNN_TAM_TrainableWithPrior
from src.models.gnn.gnn_trainable_partial_freeze import GNN_TAM_PartialFreeze


# -------------------------------------------------
# Argument parser
# -------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Unified Trainer (MLP + GNN variants)"
    )

    # data
    p.add_argument("--dataset", type=str, default="reinartz_tep")
    p.add_argument("--window_size", type=int, default=100)
    p.add_argument("--step_size", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--feature_mode", type=str, default="te_33",
                   choices=["te_33", "drop_29_41"])
    p.add_argument("--train_percent", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=0)

    # model
    p.add_argument("--model_type", type=str, default="mlp",
                   choices=["mlp", "gnn_fixed",
                            "gnn_trainable",
                            "gnn_trainable_prior",
                            "gnn_partial"])

    # adjacency
    p.add_argument("--adj_path", type=str, default=None)
    p.add_argument("--prior_path", type=str, default=None)

    # GNN shared
    p.add_argument("--n_gnn", type=int, default=1)
    p.add_argument("--n_hidden", type=int, default=1024)
    p.add_argument("--gsl_type", type=str, default="relu")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--k", type=int, default=None)

    # prior options
    p.add_argument("--train_residual", action="store_true")

    # training
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)

    # saving
    p.add_argument("--save_dir", type=str, default="outputs/runs")

    return p.parse_args()


# -------------------------------------------------
# Utilities
# -------------------------------------------------
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_feature_mode(dataset, mode):
    if mode == "te_33":
        cols = [f"xmeas_{i}" for i in range(1, 23)] + \
               [f"xmv_{i}" for i in range(1, 12)]
        dataset.df = dataset.df[cols]

    elif mode == "drop_29_41":
        start_col = 22
        end_col = min(41, dataset.df.shape[1])
        if dataset.df.shape[1] > start_col:
            dataset.df = dataset.df.drop(
                columns=dataset.df.columns[start_col:end_col]
            )


def subsample_train(dataset, percent, seed):
    if percent >= 100:
        return

    rng = np.random.default_rng(seed)
    mask = dataset.train_mask.astype(bool)
    idx = mask[mask].index.to_numpy()

    n_keep = max(1, int(len(idx) * percent / 100))
    keep = rng.choice(idx, size=n_keep, replace=False)

    new_mask = mask.copy()
    new_mask[:] = False
    new_mask.loc[keep] = True
    dataset.train_mask = new_mask


def normalize(dataset):
    scaler = StandardScaler()
    scaler.fit(dataset.df[dataset.train_mask])
    dataset.df[:] = scaler.transform(dataset.df)


def adapt_input(ts, model_type):
    if model_type in ["gnn_fixed",
                      "gnn_trainable",
                      "gnn_trainable_prior",
                      "gnn_partial"]:
        return ts.transpose(1, 2)
    return ts


def load_adjacency(path):
    if path.endswith(".npy"):
        return np.load(path)
    return np.loadtxt(path)


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = FDDDataset(name=args.dataset)

    apply_feature_mode(dataset, args.feature_mode)
    subsample_train(dataset, args.train_percent, args.seed)
    normalize(dataset)

    n_nodes = dataset.df.shape[1]
    n_classes = len(set(dataset.label))

    train_dl = FDDDataloader(
        dataframe=dataset.df,
        label=dataset.label,
        mask=dataset.train_mask,
        window_size=args.window_size,
        step_size=args.step_size,
        use_minibatches=True,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # ------------------ Build model ------------------

    if args.model_type == "mlp":
        model = MLPBaseline(args.window_size, n_nodes, n_classes)

    elif args.model_type == "gnn_fixed":
        A = load_adjacency(args.adj_path)
        model = GNN_TAM_FixedAdj(
            n_nodes, args.window_size, n_classes,
            adj_fixed=A,
            n_gnn=args.n_gnn,
            n_hidden=args.n_hidden,
            device=device,
        )

    elif args.model_type == "gnn_trainable":
        model = GNN_TAM_Trainable(
            n_nodes, args.window_size, n_classes,
            n_gnn=args.n_gnn,
            gsl_type=args.gsl_type,
            n_hidden=args.n_hidden,
            alpha=args.alpha,
            k=args.k,
            device=device,
        )

    elif args.model_type == "gnn_trainable_prior":
        A_np = load_adjacency(args.prior_path)
        A_prior = torch.tensor(A_np, dtype=torch.float32, device=device)

        model = GNN_TAM_TrainableWithPrior(
            n_nodes, args.window_size, n_classes,
            n_gnn=args.n_gnn,
            n_hidden=args.n_hidden,
            k=args.k,
            device=device,
            A_prior=A_prior,
            train_residual=args.train_residual,
        )

    elif args.model_type == "gnn_partial":
        A_np = load_adjacency(args.prior_path)
        A_prior = torch.tensor(A_np, dtype=torch.float32, device=device)

        # 🔒 freeze strong edges (customizable)
        freeze_mask = (A_prior > 0.5).float()

        model = GNN_TAM_PartialFreeze(
            n_nodes, args.window_size, n_classes,
            n_gnn=args.n_gnn,
            n_hidden=args.n_hidden,
            k=args.k,
            device=device,
            A_prior=A_prior,
            freeze_mask=freeze_mask,
        )

    else:
        raise NotImplementedError

    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    weight = torch.ones(n_classes, device=device) * 0.5
    if n_classes > 1:
        weight[1:] /= (n_classes - 1)

    # ------------------ Training ------------------

    history = {"train_loss": []}

    for e in trange(args.n_epochs):
        model.train()
        losses = []

        for ts_np, _, y_np in train_dl:
            ts = torch.as_tensor(ts_np, dtype=torch.float32, device=device)
            y = torch.as_tensor(y_np, dtype=torch.long, device=device)

            ts = adapt_input(ts, args.model_type)

            logits = model(ts)
            loss = F.cross_entropy(logits, y, weight=weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        avg_loss = float(np.mean(losses))
        history["train_loss"].append(avg_loss)
        print(f"Epoch {e+1}: {avg_loss:.4f}")

    # ------------------ Save ------------------

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = (
        f"{args.model_type}"
        f"_tp{args.train_percent:g}"
        f"_ws{args.window_size}"
        f"_seed{args.seed}"
        f"_ep{args.n_epochs}"
        f"_{timestamp}"
    )

    run_dir = Path(args.save_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model, run_dir / "model.pt")

    (run_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (run_dir / "metrics.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Saved to:", run_dir)


if __name__ == "__main__":
    main()
