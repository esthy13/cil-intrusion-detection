import torch
import random
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from src.metrics import accuracy, macro_f1
from src.task_builder import build_task

class ReservoirBuffer:
    def __init__(self, size):
        self.size = size
        self.indices = []
        self.labels = []
        self.logits = []
        self.n_seen = 0

    def add(self, indices, labels, logits):
        """
        indices: list[int]
        labels: tensor [B]
        logits: tensor [B, C]
        """
        indices = indices.tolist() if torch.is_tensor(indices) else indices
        labels = labels.detach().cpu()
        logits = logits.detach().cpu()

        for idx, y, logit in zip(indices, labels, logits):
            self.n_seen += 1

            if len(self.indices) < self.size:
                self.indices.append(idx)
                self.labels.append(y)
                self.logits.append(logit)
            else:
                j = random.randint(0, self.n_seen - 1)
                if j < self.size:
                    self.indices[j] = idx
                    self.labels[j] = y
                    self.logits[j] = logit

    def sample(self, batch_size, current_n_classes):
        if len(self.indices) == 0:
            return None

        idxs = np.random.choice(
            len(self.indices),
            min(batch_size, len(self.indices)),
            replace=False
        )

        indices = [self.indices[i] for i in idxs]
        labels = torch.stack([self.labels[i] for i in idxs])

        padded_logits = []
        for i in idxs:
            logit = self.logits[i]

            if logit.shape[0] < current_n_classes:
                pad = torch.zeros(
                    current_n_classes - logit.shape[0]
                )
                logit = torch.cat([logit, pad], dim=0)

            padded_logits.append(logit)

        logits = torch.stack(padded_logits)

        return indices, labels, logits

def train_task(model, loader, reservoir_buffer, optimizer, device,
               alpha=0.5, beta=0.5, epochs=1,):

    ce = nn.CrossEntropyLoss()
    model.train()

    for _ in range(epochs):
        for batch_idx, (x, y) in enumerate(loader):

            x, y = x.to(device), y.to(device)

            # Forward current batch
            logits, _ = model(x)
            loss = ce(logits, y)

            # ----- DER++ Replay -----
            buf = reservoir_buffer.sample(len(x), model.classifier.out_features)

            if buf is not None:
                replay_indices, replay_labels, replay_logits = buf

                # Re-fetch from BASE dataset (global indices)
                bx = torch.stack([
                    loader.dataset.dataset[i][0]
                    for i in replay_indices
                ]).to(device)

                by = replay_labels.to(device)
                blog = replay_logits.to(device)

                replay_out, _ = model(bx)

                # Expand stored logits if classifier grew
                if blog.shape[1] < replay_out.shape[1]:
                    pad = torch.zeros(
                        blog.shape[0],
                        replay_out.shape[1] - blog.shape[1],
                        device=device
                    )
                    blog = torch.cat([blog, pad], dim=1)

                # DER++
                loss += alpha * F.mse_loss(replay_out, blog)
                loss += beta * ce(replay_out, by)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # ----- Add current batch to buffer -----
            original_indices = [
                loader.dataset.indices[i]
                for i in range(
                    batch_idx * loader.batch_size,
                    batch_idx * loader.batch_size + len(x)
                )
            ]

            reservoir_buffer.add(original_indices, y, logits)

def evaluate(model, dataset, seen_classes, device, batch_size=256):
    model.eval()

    eval_dataset = build_task(dataset, seen_classes)
    loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits, _ = model(x)
            preds = logits.argmax(1).cpu().numpy()

            all_preds.append(preds)
            all_targets.append(y.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    acc = accuracy(all_targets, eval_dataset.y)
    f1  = macro_f1(all_targets, eval_dataset.y)

    return acc, f1, all_targets, all_preds
