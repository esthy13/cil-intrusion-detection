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

#Class from Margarita
class IDSIncrementalDataset(Dataset):
    def __init__(self, df, feature_cols, label_col, label_to_id, indices=None):

        self.label_to_id = label_to_id

        self.all_features = df[feature_cols].values.astype(np.float32)
        self.all_labels = df[label_col].map(label_to_id).values.astype(np.int64)
        self.all_indices = np.arange(len(self.all_labels))

        # Initial state
        self.indices = self.all_indices
        self.features = self.all_features[self.indices]
        self.labels = self.all_labels[self.indices]

        # Up-to normalization stats
        self.min_up_to = None
        self.max_up_to = None

        # Memory
        self.memory_ids = []

        # total labels
        self.total_labels = np.max(self.all_labels)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        id_ = self.indices[idx]
        x = self.features[idx]
        y = self.labels[idx]

        return id_, x, y

    # filter by class and return ids (dataset ORIGINAL)
    def filter_by_classes(self, class_list):
        mask = np.isin(self.all_labels, class_list)
        new_indices = self.all_indices[mask]
        return new_indices

    # Build task dataset (new + memory)
    def build_task_dataset(self, new_class_list):

        # new class ids
        new_classes_indices = self.filter_by_classes(new_class_list)

        if len(self.memory_ids) == 0:
            total_indices = new_classes_indices
        else:
            total_indices = np.concatenate([self.memory_ids, new_classes_indices])

        # Update dataset view
        self.indices = total_indices
        self.labels = self.all_labels[self.indices]
        self.features = self.all_features[self.indices]

        # Up-to normalization (task dataset)
        self.compute_up_to_stats()
        self.apply_min_max_normalization()

    def apply_min_max_normalization(self):
        if self.min_up_to is None or self.max_up_to is None:
            raise ValueError(
                "Up-to stats not computed. Call compute_up_to_stats() first."
            )

        eps = 1e-8  # Avoid Zero error division

        self.features = (self.features - self.min_up_to) / (
            self.max_up_to - self.min_up_to + eps
        )

        self.features = self.features.astype(np.float32)

    def update_memory(self, memory_ids):
        self.memory_ids = memory_ids

    # compute stats over build task dataset (new + memory)
    def compute_up_to_stats(self):
        self.max_up_to = self.features.max(axis=0)
        self.min_up_to = self.features.min(axis=0)
