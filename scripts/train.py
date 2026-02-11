import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm, trange

from fddbenchmark import FDDDataset, FDDDataloader

# ---- models ----
from src.models.baselines.mlp import MLPBaseline


# -------------------------------------------------
# Argument parser
# -------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Unified trainer (train/test only)")

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
                   choices=["mlp", "gru", "cnn1d", "gnn"])

    # training
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)

    # saving
    p.add_argument("--save_dir", type=str, default="outputs/runs")
    p.add_argument("--run_name", type=str, default=None)

    return p.parse_args()


# -------------------------------------------------
# Utilities
# -------------------------------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_feature_mode(dataset: FDDDataset, mode: str):
    if mode == "te_33":
        cols = [f"xmeas_{i}" for i in range(1, 23)] + [f"xmv_{i}" for i in range(1, 12)]
        missing = [c for c in cols if c not in dataset.df.columns]
        if missing:
            raise ValueError(f"Missing TE columns: {missing}")
        dataset.df = dataset.df[cols]
        return

    if mode == "drop_29_41":
        start_col = 22
        end_col = min(41, dataset.df.shape[1])
        if dataset.df.shape[1] > start_col:
            dataset.df = dataset.df.drop(columns=dataset.df.columns[start_col:end_col])
        return

    raise ValueError(f"Unknown feature_mode: {mode}")


def subsample_train(dataset: FDDDataset, percent: float, seed: int):
    if percent >= 100:
        return
    if percent <= 0:
        raise ValueError("train_percent must be > 0")

    rng = np.random.default_rng(seed)
    mask = dataset.train_mask.astype(bool)
    idx = mask[mask].index.to_numpy()

    n_keep = max(1, int(len(idx) * percent / 100))
    keep = rng.choice(idx, size=n_keep, replace=False)

    new_mask = mask.copy()
    new_mask[:] = False
    new_mask.loc[keep] = True
    dataset.train_mask = new_mask


def normalize(dataset: FDDDataset):
    scaler = StandardScaler()
    scaler.fit(dataset.df[dataset.train_mask])
    dataset.df[:] = scaler.transform(dataset.df)


def adapt_input(ts: torch.Tensor, model_type: str) -> torch.Tensor:
    # ts: [B, T, F]
    if model_type in ["cnn1d", "gnn"]:
        return ts.transpose(1, 2)  # [B, F, T]
    return ts


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------
    # 1) Load dataset
    # ------------------
    dataset = FDDDataset(name=args.dataset)

    apply_feature_mode(dataset, args.feature_mode)
    subsample_train(dataset, args.train_percent, args.seed)
    normalize(dataset)

    n_nodes = dataset.df.shape[1]
    n_classes = len(set(dataset.label))

    print(f"n_nodes = {n_nodes}, n_classes = {n_classes}")

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

    # ------------------
    # 2) Build model
    # ------------------
    if args.model_type == "mlp":
        model = MLPBaseline(
            window_size=args.window_size,
            n_nodes=n_nodes,
            n_classes=n_classes,
        )
    else:
        raise NotImplementedError("Only MLP is wired for now.")

    model.to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    weight = torch.ones(n_classes, device=device) * 0.5
    if n_classes > 1:
        weight[1:] /= (n_classes - 1)

    # ------------------
    # 3) Run directory
    # ------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name
    if run_name is None:
        run_name = f"{args.model_type}_tp{args.train_percent:g}_ws{args.window_size}_seed{args.seed}_{timestamp}"

    run_dir = Path(args.save_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # ------------------
    # 4) Training loop
    # ------------------
    history = {"train_loss": []}

    for e in trange(args.n_epochs, desc="Epochs"):
        model.train()
        losses = []

        for ts_np, _, y_np in tqdm(train_dl, leave=False):
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
        print(f"Epoch {e+1:02d}/{args.n_epochs} | train_loss = {avg_loss:.4f}")

    # ------------------
    # 5) Save model
    # ------------------
    torch.save(model, run_dir / "model.pt")
    (run_dir / "metrics.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Model saved to: {run_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
