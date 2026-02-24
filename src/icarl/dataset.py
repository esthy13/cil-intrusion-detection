import numpy as np
import pandas as pd
import zipfile
import torch
from torch.utils.data import Dataset

def load_csv_folder_from_zip(path_zip, folder):
    dfs = []

    with zipfile.ZipFile(path_zip, 'r') as z:
        csv_files = sorted([
            name for name in z.namelist()
            if name.startswith(folder + "/") and name.endswith(".csv")
        ])

        if not csv_files:
            raise ValueError(f"No CSV files found in {folder}/ inside {path_zip}")

        for file in csv_files:
            with z.open(file) as f:
                dfs.append(pd.read_csv(f))

    return pd.concat(dfs, ignore_index=True)

class TaskDatasetView(Dataset):
    """
    Lightweight dataset view over a subset of indices.
    Used for:
    - Task dataset (new + memory) -> Cross Entropy
    - Memory dataset (only memory) -> Knowledge Distillation
    """
    def __init__(self, all_features, all_labels, indices, statistics, eps=1e-8):
        super().__init__()
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

class CILDataset:
    def __init__(self, path_zip, dataset_name, split, eps=1e-8):

        """
        path_folder: path to dataset folder
        split: 'train' or 'test'
        """
        df = load_csv_folder_from_zip(f'{path_zip}/{dataset_name}.zip', f'{dataset_name}/{split}')

        self.label_col_name = None
        self.label_to_id = None
        if dataset_name == 2015:
            self.label_to_id = {'Normal': 0,
                                'Analysis': 1,
                                'Backdoor': 2,
                                'DoS': 3,
                                'Exploits': 4,
                                'Fuzzers': 5,
                                'Generic': 6,
                                'Reconnaissance': 7,
                                'Shellcode': 8}
            self.label_col_name = 'attack_cat'
        elif dataset_name == 2017:
            self.label_to_id = {'benign': 0,
                                'bot': 1,
                                'ddos': 2,
                                'dos': 3,
                                'ftp-patator': 4,
                                'portscan': 5,
                                'ssh-patator': 6,
                                'web-attack': 7}
            self.label_col_name = 'Label'


        # Global immutable dataset
        self.all_features = df.drop(columns=[self.label_col_name]).values.astype(np.float32)
        self.all_labels = df[self.label_col_name].map(self.label_to_id).values.astype(np.int64)
        self.all_indices = np.arange(len(self.all_labels), dtype=np.int64)
        self.input_dim = self.all_features.shape[1]

        # Memory buffer (global indices)
        self.memory_indices = np.array([], dtype=np.int64)

        # Up-to normalization stats (per feature)
        self.min_up_to = None
        self.max_up_to = None

        # Precompute denominator for speed
        self.eps = eps

        # Total labels
        self.total_labels = int(np.max(self.all_labels)) + 1

    def __getitem__(self, idx):
        real_idx = self.all_indices[idx]
        x = self.all_features[real_idx]
        x = self.apply_min_max_up_to_normalization(x)
        y = self.all_labels[real_idx]

        # Convert to torch tensors
        x = torch.from_numpy(x)
        y = torch.as_tensor(y, dtype=torch.long)

        return real_idx, x, y

    def __len__(self):
        return len(self.all_indices)

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

    def apply_min_max_up_to_normalization(self, x):

        self.denominator = np.maximum((self.max_up_to - self.min_up_to),self.eps)

        return ((x - self.min_up_to) / self.denominator).astype(np.float32)

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

    def build_task_testset(self,task_labels, stats):
        testdatasets = []
        for i in range(len(task_labels)):
            task_labels_ids = self.filter_by_classes(task_labels[i])
            testdatasets.append(TaskDatasetView(
            self.all_features,
            self.all_labels,
            task_labels_ids,
            stats))

        return testdatasets
