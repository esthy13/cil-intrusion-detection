import torch
import numpy as np
from torch.utils.data import Dataset

class RemappedSubset(Dataset):
    """
    Subset that remaps global class indices to [0..C-1]
    """
    def __init__(self, dataset, indices, class_ids):
        self.dataset = dataset
        self.indices = indices
        self.class_map = {cid: i for i, cid in enumerate(class_ids)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        return x, torch.tensor(self.class_map[y.item()])

def build_task(dataset, class_names):
    class_ids = [dataset.class_to_idx[c] for c in class_names]
    idxs = np.where(np.isin(dataset.y, class_ids))[0]
    return RemappedSubset(dataset, idxs, class_ids)

def build_scenario( all_classes, attacks_pattern, benign_class="benign"):
    """
    all_classes: ordered list of class names (benign must be first)
    attacks_pattern: list of ints, number of NEW attacks per task
                     e.g. [1,1,1] or [3,2] or [5]
    benign_class: name of benign class (default: 'benign')

    Returns:
        tasks: list of lists of class names (cumulative)
    """

    if benign_class not in all_classes:
        raise ValueError(f"Benign class '{benign_class}' not found in classes")

    if all_classes[0] != benign_class:
        raise ValueError(
            f"Benign class must be index 0, got {all_classes[0]}"
        )

    attack_classes = [c for c in all_classes if c != benign_class]

    if sum(attacks_pattern) != len(attack_classes):
        raise ValueError(
            f"Invalid attacks_pattern: sum={sum(attacks_pattern)}, "
            f"but there are {len(attack_classes)} attack classes"
        )

    tasks = []
    current_index = 0

    for _, n_new in enumerate(attacks_pattern):
        current_index += n_new
        seen_attacks = attack_classes[:current_index]
        seen_classes = [benign_class] + seen_attacks
        tasks.append(seen_classes)

    return tasks, attacks_pattern

class UpToNormalizer:
    """
    Continual min-max normalizer using only past and present data
    """
    def __init__(self):
        self.min = None
        self.max = None

    def update(self, x):
        """
        x: numpy array [N, D]
        """
        batch_min = x.min(axis=0)
        batch_max = x.max(axis=0)

        if self.min is None:
            self.min = batch_min
            self.max = batch_max
        else:
            self.min = np.minimum(self.min, batch_min)
            self.max = np.maximum(self.max, batch_max)

    def normalize(self, x):
        """
        x: numpy array [N, D]
        """
        eps = 1e-8
        return (x - self.min) / (self.max - self.min + eps)
