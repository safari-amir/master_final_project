import numpy as np
from sklearn.preprocessing import StandardScaler

from fddbenchmark import FDDDataset, FDDDataloader


class TEPDataModule:
    """
    Data handler for Tennessee Eastman Process (TEP)
    based on fddbenchmark, with controllable train percentage.
    """

    def __init__(
        self,
        dataset_name: str = "reinartz_tep",
        window_size: int = 100,
        step_size: int = 1,
        batch_size: int = 512,
        train_percent: float = 100.0,
        seed: int = 0,
    ):
        self.dataset_name = dataset_name
        self.window_size = window_size
        self.step_size = step_size
        self.batch_size = batch_size
        self.train_percent = train_percent
        self.seed = seed

        self.dataset = None
        self.scaler = None

    def prepare(self):
        """
        Load dataset, select 33 TE variables,
        subsample train data, and normalize.
        """
        # ------------------
        # Load dataset
        # ------------------
        self.dataset = FDDDataset(name=self.dataset_name)

        # ------------------
        # Select 33 TE variables (CIPCaD standard)
        # ------------------
        cols_te_33 = (
            [f"xmeas_{i}" for i in range(1, 23)] +
            [f"xmv_{i}" for i in range(1, 12)]
        )

        missing = [c for c in cols_te_33 if c not in self.dataset.df.columns]
        if missing:
            raise ValueError(f"Missing TE columns: {missing}")

        self.dataset.df = self.dataset.df[cols_te_33]

        # ------------------
        # Subsample train data
        # ------------------
        if self.train_percent < 100.0:
            if self.train_percent <= 0:
                raise ValueError("train_percent must be > 0")

            rng = np.random.default_rng(self.seed)
            train_mask = self.dataset.train_mask.astype(bool)
            train_indices = train_mask[train_mask].index.to_numpy()

            n_keep = max(1, int(len(train_indices) * self.train_percent / 100.0))
            keep_idx = rng.choice(train_indices, size=n_keep, replace=False)

            new_mask = train_mask.copy()
            new_mask[:] = False
            new_mask.loc[keep_idx] = True
            self.dataset.train_mask = new_mask

        # ------------------
        # Normalize (fit on train only)
        # ------------------
        self.scaler = StandardScaler()
        self.scaler.fit(self.dataset.df[self.dataset.train_mask])
        self.dataset.df[:] = self.scaler.transform(self.dataset.df)

    def train_dataloader(self):
        return FDDDataloader(
            dataframe=self.dataset.df,
            label=self.dataset.label,
            mask=self.dataset.train_mask,
            window_size=self.window_size,
            step_size=self.step_size,
            use_minibatches=True,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return FDDDataloader(
            dataframe=self.dataset.df,
            label=self.dataset.label,
            mask=self.dataset.val_mask,
            window_size=self.window_size,
            step_size=self.step_size,
            use_minibatches=True,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return FDDDataloader(
            dataframe=self.dataset.df,
            label=self.dataset.label,
            mask=self.dataset.test_mask,
            window_size=self.window_size,
            step_size=self.step_size,
            use_minibatches=True,
            batch_size=self.batch_size,
            shuffle=False,
        )
