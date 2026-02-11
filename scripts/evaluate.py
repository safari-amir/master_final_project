import argparse
from pathlib import Path
from contextlib import redirect_stdout

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from fddbenchmark import FDDDataset, FDDDataloader, FDDEvaluator


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate saved model with fddbenchmark FDDEvaluator")

    p.add_argument("--dataset", type=str, default="reinartz_tep")
    p.add_argument("--window_size", type=int, default=100)
    p.add_argument("--step_size", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=512)

    p.add_argument("--feature_mode", type=str, default="drop_29_41",
                   choices=["drop_29_41", "te_33"])

    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--model_type", type=str, required=True,
                   choices=["mlp", "cnn1d", "gru", "gnn"])

    p.add_argument("--eval_split", type=str, default="test",
                   choices=["test", "val", "train"])
    p.add_argument("--evaluator_step_size", type=int, default=1)

    return p.parse_args()


def apply_feature_mode(dataset: FDDDataset, mode: str):
    if mode == "drop_29_41":
        start_col = 22
        end_col = min(41, dataset.df.shape[1])
        if dataset.df.shape[1] > start_col:
            dataset.df = dataset.df.drop(columns=dataset.df.columns[start_col:end_col])
        return

    if mode == "te_33":
        cols = [f"xmeas_{i}" for i in range(1, 23)] + [f"xmv_{i}" for i in range(1, 12)]
        dataset.df = dataset.df[cols]
        return

    raise ValueError(mode)


def normalize_like_paper(dataset: FDDDataset):
    scaler = StandardScaler()
    scaler.fit(dataset.df[dataset.train_mask])
    dataset.df[:] = scaler.transform(dataset.df)


def get_split_mask(dataset: FDDDataset, split: str):
    if split == "train":
        return dataset.train_mask
    if split == "val":
        return dataset.val_mask
    return dataset.test_mask


def adapt_input(ts: torch.Tensor, model_type: str) -> torch.Tensor:
    if model_type in ["cnn1d", "gnn"]:
        return ts.transpose(1, 2)
    return ts


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------
    # Data preparation
    # ------------------
    dataset = FDDDataset(name=args.dataset)
    apply_feature_mode(dataset, args.feature_mode)
    normalize_like_paper(dataset)

    mask = get_split_mask(dataset, args.eval_split)

    dl = FDDDataloader(
        dataframe=dataset.df,
        label=dataset.label,
        mask=mask,
        window_size=args.window_size,
        step_size=args.step_size,
        use_minibatches=True,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # ------------------
    # Load model
    # ------------------
    model_path = Path(args.model_path)
    model = torch.load(model_path, map_location=device)
    model.eval()

    run_dir = model_path.parent
    print(f"Saving evaluation metrics to: {run_dir}")

    # ------------------
    # Inference
    # ------------------
    preds, labels_all = [], []

    with torch.no_grad():
        for ts_np, index, y_np in dl:
            ts = torch.as_tensor(ts_np, dtype=torch.float32, device=device)
            ts = adapt_input(ts, args.model_type)

            logits = model(ts)
            pred = logits.argmax(dim=1).cpu().numpy()

            preds.append(pd.Series(pred, index=index))
            labels_all.append(pd.Series(y_np, index=index))

    pred_ser = pd.concat(preds).sort_index()
    label_ser = pd.concat(labels_all).sort_index()

    # ------------------
    # Evaluation (save only metrics)
    # ------------------
    evaluator = FDDEvaluator(step_size=args.evaluator_step_size)

    with open(run_dir / "eval_metrics.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            evaluator.print_metrics(label_ser, pred_ser)

    # also print to console
    evaluator.print_metrics(label_ser, pred_ser)


if __name__ == "__main__":
    main()
