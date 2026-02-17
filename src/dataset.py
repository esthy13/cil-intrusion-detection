import torch
import os
import glob
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

def trial():
    print("This is a trail method")

# dataset wrapper
class IDSBaseDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        """
        root_dir: path to 2017/
        split: 'train' or 'test'
        """
        csv_dir = os.path.join(root_dir, split)
        csvs = glob.glob(os.path.join(csv_dir, "*.csv"))
        assert len(csvs) > 0, f"No CSV files found in {csv_dir}"

        df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)

        labels = list(df["Label"].unique())

        if "benign" not in labels:
            raise ValueError("Dataset must contain a 'benign' class")

        # Enforcing benign as class 0
        labels = ["benign"] + sorted([l for l in labels if l != "benign"])

        self.classes = labels
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.x = df.drop(columns=["Label"]).values.astype(np.float32)
        self.y = np.array(
            [self.class_to_idx[label] for label in df["Label"]],
            dtype=np.int64
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])
    def set_features(self, new_x):
        assert new_x.shape == self.x.shape
        self.x = new_x.astype(np.float32)