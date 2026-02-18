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
import numpy as np
import torch
from torch.utils.data import Dataset

class TaskDatasetView(Dataset):
    """
    Lightweight dataset view over a subset of indices.
    Used for:
    - Task dataset (new + memory) -> Cross Entropy
    - Memory dataset (only memory) -> Knowledge Distillation
    """
    def __init__(self, all_features, all_labels, indices, statistics, eps=1e-8):
        self.features = all_features
        self.labels = all_labels
        self.indices = np.array(indices, dtype=np.int64)

        self.min_up_to = np.asarray(statistics['min_up_to'], dtype=np.float32)
        self.max_up_to = np.asarray(statistics['max_up_to'], dtype=np.float32)

        # Precompute denominator for speed
        self.eps = eps
        self.denominator = np.maximum((self.max_up_to - self.min_up_to),self.eps)
    
    def apply_min_max_up_to_normalization(self, x):

        return ((x - self.min_up_to) / self.denominator).astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x = self.features[real_idx]
        x = self.apply_min_max_up_to_normalization(x)
        y = self.labels[real_idx]

        # Convert to torch tensors
        x = torch.from_numpy(x)
        y = torch.as_tensor(y, dtype=torch.long)
        
        return real_idx, x, y


class IDSIncrementalDataset:
    def __init__(self, root_dir, split, feature_cols, label_col, label_to_id):

        """
        root_dir: path to 2017/
        split: 'train' or 'test'
        """
        csv_dir = os.path.join(root_dir, split)
        csvs = glob.glob(os.path.join(csv_dir, "*.csv"))
        assert len(csvs) > 0, f"No CSV files found in {csv_dir}"

        df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)

        self.label_to_id = label_to_id

        # Global immutable dataset
        self.all_features = df[feature_cols].values.astype(np.float32)
        self.all_labels = df[label_col].map(label_to_id).values.astype(np.int64)
        self.all_indices = np.arange(len(self.all_labels), dtype=np.int64)

        # Memory buffer (global indices)
        self.memory_indices = np.array([], dtype=np.int64)

        # Up-to normalization stats (per feature)
        self.min_up_to = None
        self.max_up_to = None

        # Total labels (fix torch.max bug)
        self.total_labels = int(np.max(self.all_labels)) + 1

    def filter_by_classes(self, class_list):
        mask = np.isin(self.all_labels, class_list)
        return self.all_indices[mask]

    def compute_up_to_stats(self, seen_indices):
        """
        Compute cumulative min/max over ALL seen samples (global indices).
        """
        seen_features = self.all_features[seen_indices]

        current_min = seen_features.min(axis=0)
        current_max = seen_features.max(axis=0)

        if self.min_up_to is None:
            self.min_up_to = current_min
            self.max_up_to = current_max
        else:
            self.min_up_to = np.minimum(self.min_up_to, current_min)
            self.max_up_to = np.maximum(self.max_up_to, current_max)

    def update_memory(self, memory_ids):
        """
        memory_ids must be GLOBAL indices.
        """
        if memory_ids is None or len(memory_ids) == 0:
            self.memory_indices = np.array([], dtype=np.int64)
        else:
            self.memory_indices = np.array(memory_ids, dtype=np.int64)

    def build_task_dataset(self, new_class_list):
        """
        Returns:
        - task_dataset (memory + new) -> Cross Entropy
        - memory_dataset (memory only) -> Knowledge Distillation
        """

        # Get global indices of new classes
        new_indices = self.filter_by_classes(new_class_list)

        # Combine memory + new (GLOBAL indices)
        if len(self.memory_indices) == 0:
            total_indices = new_indices
        else:
            total_indices = np.concatenate([self.memory_indices, new_indices])

        # Compute true up-to stats (over all seen samples)
        self.compute_up_to_stats(total_indices)

        stats = {
            'min_up_to': self.min_up_to,
            'max_up_to': self.max_up_to
        }

        # IMPORTANT: pass ALL features + GLOBAL indices
        task_dataset = TaskDatasetView(
            self.all_features,
            self.all_labels,
            total_indices,
            stats
        )

        # Memory dataset for KD
        if len(self.memory_indices) > 0:
            memory_dataset = TaskDatasetView(
                self.all_features,
                self.all_labels,
                self.memory_indices,
                stats
            )
        else:
            memory_dataset = None

        return task_dataset, memory_dataset
