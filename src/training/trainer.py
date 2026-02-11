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

# ---- your models (adjust if your imports differ) ----
# MLP we already wrote:
from src.models.baselines.mlp import MLPBaseline

# If you already have these, uncomment + fix paths:
# from src.models.baselines.gru import GRUBaseline
# from src.models.baselines.cnn1d import CNN1DBaseline
# from src.models.gnn.gnn_trainable_adj import GNN_TAM


def parse_args():
    p = argparse.ArgumentParser(description="Unified trainer (fddbenchmark)")

    # data
    p.add_argument("--dataset", type=str, default="reinartz_tep")
    p.add_argument("--window_size", type=int, default=100)
    p.add_argument("--step_size", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--feature_mode", type=str, default="te_33",
                   choices=["te_33", "drop_29_41"],
                   help="te_33: select xmeas_1..22 + xmv_1..11 | drop_29_41: drop cols 29..41 (paper code style)")
    p.add_argument("--train_percent", type=float, default=100.0,
                   help="percentage of train data to use (0..100]")
    p.add_argument("--seed", type=int, default=0)

    # model
    p.add_argument("--model_type", type=str, required=True,
                   choices=["mlp", "gru", "cnn1d", "gnn"])
    # common training
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save_name", type=str, default=None,
                   help="Optional run name; if not provided, auto-generated.")
    p.add_argument("--save_dir", type=str, default="outputs/runs")

    # (Optional) evaluate quickly after training (prints only)
    p.add_argument("--quick_eval_split", type=str, default=None,
                   choices=[None, "val", "test"],
                   help="If set, will do a quick evaluation pass and print accuracy only (not FDDEvaluator).")

    # -------- GNN args (kept for later) --------
    p.add_argument("--n_gnn", type=int, default=1)
    p.add_argument("--gsl_type", type=str, default="relu",
                   choices=["relu", "directed", "unidirected", "undirected"])
    p.add_argument("--n_hidden", type=int, default=1024)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--k", type=int, default=None)

    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_feature_mode(dataset: FDDDataset, mode: str):
    if mode == "te_33":
        cols_te_33 = [f"xmeas_{i}" for i in range(1, 23)] + [f"xmv_{i}" for i in range(1, 12)]
        missing = [c for c in cols_te_33 if c not in dataset.df.columns]
        if missing:
            raise ValueError(f"Missing columns for te_33: {missing}")
        dataset.df = dataset.df[cols_te_33]
        return

    if mode == "drop_29_41":
        # مانند کد مقاله شما
        n_features_total = dataset.df.shape[1]
        start_col = 22  # مطابق کد شما
        end_col = 41
        if n_features_total > start_col:
            end_col = min(end_col, n_features_total)
            cols_to_drop = dataset.df.columns[start_col:end_col]
            dataset.df = dataset.df.drop(columns=cols_to_drop)
        return

    raise ValueError(f"Unknown feature_mode: {mode}")


def subsample_train_mask(dataset: FDDDataset, train_percent: float, seed: int):
    if train_percent >= 100.0:
        return

    if train_percent <= 0.0:
        raise ValueError("train_percent must be in (0, 100].")

    rng = np.random.default_rng(seed)
    train_mask = dataset.train_mask.astype(bool)
    train_indices = train_mask[train_mask].index.to_numpy()

    n_keep = max(1, int(len(train_indices) * train_percent / 100.0))
    keep_idx = rng.choice(train_indices, size=n_keep, replace=False)

    new_mask = train_mask.copy()
    new_mask[:] = False
    new_mask.loc[keep_idx] = True

    dataset.train_mask = new_mask


def normalize_fit_on_train(dataset: FDDDataset):
    scaler = StandardScaler()
    scaler.fit(dataset.df[dataset.train_mask])
    dataset.df[:] = scaler.transform(dataset.df)
    return scaler


def build_dataloaders(dataset: FDDDataset, window_size: int, step_size: int, batch_size: int):
    train_dl = FDDDataloader(
        dataframe=dataset.df,
        label=dataset.label,
        mask=dataset.train_mask,
        window_size=window_size,
        step_size=step_size,
        use_minibatches=True,
        batch_size=batch_size,
        shuffle=True
    )

    val_dl = FDDDataloader(
        dataframe=dataset.df,
        label=dataset.label,
        mask=dataset.val_mask,
        window_size=window_size,
        step_size=step_size,
        use_minibatches=True,
        batch_size=batch_size,
        shuffle=False
    )

    test_dl = FDDDataloader(
        dataframe=dataset.df,
        label=dataset.label,
        mask=dataset.test_mask,
        window_size=window_size,
        step_size=step_size,
        use_minibatches=True,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dl, val_dl, test_dl


def adapt_input(ts: torch.Tensor, model_type: str) -> torch.Tensor:
    """
    ts from FDDDataloader: [B, T, F]
    """
    if model_type == "gnn":
        return ts.transpose(1, 2)  # [B, F, T]
    if model_type == "cnn1d":
        return ts.transpose(1, 2)  # common CNN1D: [B, F, T]
    # mlp / gru: [B, T, F]
    return ts


def build_model(args, n_nodes: int, n_classes: int, device: torch.device):
    if args.model_type == "mlp":
        model = MLPBaseline(
            window_size=args.window_size,
            n_nodes=n_nodes,
            n_classes=n_classes,
            hidden_dims=(512, 256),
            dropout=0.2,
        )
        return model.to(device)

    # ---- placeholders: connect these when you implement them ----
    if args.model_type == "gru":
        raise NotImplementedError("GRU model not wired yet. Implement GRUBaseline and import it here.")
        # model = GRUBaseline(n_nodes=n_nodes, n_classes=n_classes, hidden_size=128, num_layers=1)
        # return model.to(device)

    if args.model_type == "cnn1d":
        raise NotImplementedError("CNN1D model not wired yet. Implement CNN1DBaseline and import it here.")
        # model = CNN1DBaseline(n_nodes=n_nodes, window_size=args.window_size, n_classes=n_classes)
        # return model.to(device)

    if args.model_type == "gnn":
        raise NotImplementedError("GNN model not wired yet. Import your GNN_TAM here and build it.")
        # model = GNN_TAM(
        #     n_nodes=n_nodes,
        #     window_size=args.window_size,
        #     n_classes=n_classes,
        #     n_gnn=args.n_gnn,
        #     gsl_type=args.gsl_type,
        #     n_hidden=args.n_hidden,
        #     alpha=args.alpha,
        #     k=args.k,
        #     device=device,
        # )
        # return model.to(device)

    raise ValueError(f"Unknown model_type: {args.model_type}")


def evaluate_simple_accuracy(model, dl, device, model_type: str):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for ts_np, _, y_np in dl:
            ts = torch.as_tensor(ts_np, dtype=torch.float32, device=device)
            y = torch.as_tensor(y_np, dtype=torch.long, device=device)
            ts = adapt_input(ts, model_type)
            logits = model(ts)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    model.train()
    return correct / max(1, total)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------
    # 1) Load dataset (fddbenchmark)
    # ------------------
    dataset = FDDDataset(name=args.dataset)

    # feature selection
    apply_feature_mode(dataset, args.feature_mode)

    # subsample train
    subsample_train_mask(dataset, args.train_percent, args.seed)

    # normalize (fit train only)
    normalize_fit_on_train(dataset)

    n_nodes = dataset.df.shape[1]
    n_classes = len(set(dataset.label))
    print(f"n_nodes={n_nodes}, n_classes={n_classes}")

    train_dl, val_dl, test_dl = build_dataloaders(
        dataset, args.window_size, args.step_size, args.batch_size
    )

    # ------------------
    # 2) Build model
    # ------------------
    model = build_model(args, n_nodes=n_nodes, n_classes=n_classes, device=device)
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # class weights (همان سبک کد مقاله)
    weight = torch.ones(n_classes, device=device) * 0.5
    if n_classes > 1:
        weight[1:] /= (n_classes - 1)

    # ------------------
    # 3) Run folder
    # ------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.save_name
    if run_name is None:
        run_name = f"{args.model_type}_fm{args.feature_mode}_tp{args.train_percent:g}_ws{args.window_size}_seed{args.seed}_{timestamp}"

    run_dir = Path(args.save_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # save config
    (run_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------
    # 4) Train loop
    # ------------------
    history = {"train_loss": [], "val_acc": []}

    for e in trange(args.n_epochs, desc="Epochs"):
        model.train()
        losses = []

        for ts_np, _, y_np in tqdm(train_dl, desc=f"Epoch {e+1}", leave=False):
            ts = torch.as_tensor(ts_np, dtype=torch.float32, device=device)  # [B, T, F]
            y = torch.as_tensor(y_np, dtype=torch.long, device=device)

            ts = adapt_input(ts, args.model_type)

            logits = model(ts)
            loss = F.cross_entropy(logits, y, weight=weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        avg_loss = float(np.mean(losses)) if losses else float("nan")
        history["train_loss"].append(avg_loss)

        # quick val accuracy each epoch (lightweight)
        val_acc = evaluate_simple_accuracy(model, val_dl, device, args.model_type)
        history["val_acc"].append(val_acc)

        print(f"Epoch {e+1:02d}/{args.n_epochs} | train_loss={avg_loss:.4f} | val_acc={val_acc:.4f}")

    # ------------------
    # 5) Save artifacts
    # ------------------
    torch.save(model, run_dir / "model.pt")
    (run_dir / "metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved model to: {run_dir / 'model.pt'}")
    print(f"Saved metrics to: {run_dir / 'metrics.json'}")

    # ------------------
    # 6) Optional quick eval
    # ------------------
    if args.quick_eval_split is not None:
        dl = val_dl if args.quick_eval_split == "val" else test_dl
        acc = evaluate_simple_accuracy(model, dl, device, args.model_type)
        print(f"Quick {args.quick_eval_split} accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
