import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import cycle
from torch.utils.data import DataLoader
from tqdm import tqdm
from icarl.metrics import accuracy, attack_accuracy, average_accuracy, compute_forgetting, save_training_results
from icarl.model import IDSNet
from icarl.dataset import CILDataset
from icarl.utils import build_task_icarl, get_device

def get_class_weights(labels, num_classes, device):
    counts = torch.bincount(labels, minlength=num_classes).float()

    # evitar división por cero (clases sin samples en el task)
    counts[counts == 0] = 1.0

    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # normalización opcional

    return weights.to(device)

class ICaRL:
    def __init__(self, model, device, memory_size=2000):
        self.model = model.to(device)
        self.device = device
        self.memory_size = memory_size

        self.seen_classes = []
        self.old_classes = []
        self.new_classes = []

        self.exemplars = {}          # class_id -> indices
        self.class_means = {}        # class_id -> mean vector
        self.old_model = None

    def add_classes(self, new_classes):
        self.new_classes = new_classes
        self.old_classes = self.seen_classes.copy()
        self.seen_classes.extend(new_classes)

        num_classes = len(self.seen_classes)
        self.model.update_classifier(num_classes)

        # Transfer old classifier weights
        if self.old_model is not None:
            old_w = self.old_model.classifier.weight.data.clone()
            self.model.classifier.weight.data[:old_w.size(0)] = old_w
            old_b = self.old_model.classifier.bias.data.clone()
            self.model.classifier.bias.data[:old_w.size(0)] = old_b

    def _distillation_loss(self, student_logits, teacher_logits, T=2.0):
        num_old = teacher_logits.shape[1]

        student = F.log_softmax(student_logits[:, :num_old] / T, dim=1)
        teacher = F.softmax(teacher_logits / T, dim=1)

        return F.kl_div(student, teacher, reduction="batchmean") * (T ** 2)

    def train_task(
        self,
        train_dataset_bce,
        train_dataset_kd=None,
        epochs=10,
        batch_size=64,
        lr=1e-3,
        lambda_kd=0.5,
        T=2,
    ):
        self.model.train()

        bce_loader = DataLoader(
            train_dataset_bce,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )

        all_labels = torch.cat([y for _, _, y in bce_loader])
        print(len(self.seen_classes))
        class_weights = get_class_weights(all_labels, len(self.seen_classes), self.device)

        if train_dataset_kd is not None and self.old_model is not None:
            kd_loader = DataLoader(
                train_dataset_kd,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2
            )
            kd_iter = cycle(kd_loader)
        else:
            kd_loader = None

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=lr * 0.01
        )

        for epoch in range(epochs):
            total_loss = 0.0

            loop = tqdm(bce_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for idx_bce, x_bce, y_bce in loop:
                x_bce = x_bce.to(self.device)
                y_bce = y_bce.to(self.device)

                # ----- Classification loss (new + memory) -----
                student_logits, _ = self.model(x_bce)
                loss_ce = F.cross_entropy(student_logits, y_bce, weight=class_weights)

                loss = loss_ce

                # ----- Distillation loss (memory only) -----
                if kd_loader is not None:
                    _, x_kd, _ = next(kd_iter)
                    x_kd = x_kd.to(self.device)

                    if self.old_model is not None:
                        self.old_model.eval()

                    with torch.no_grad():
                        teacher_logits, _ = self.old_model(x_kd)

                    student_kd_logits, _ = self.model(x_kd)
                    loss_kd = self._distillation_loss(
                        student_kd_logits,
                        teacher_logits,
                        T = T
                    )

                    loss += lambda_kd * loss_kd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            scheduler.step()
            print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(bce_loader):.4f}")

        # redefine old_model
        self.old_model = copy.deepcopy(self.model).to(self.device).eval()
        for p in self.old_model.parameters():
            p.requires_grad = False

    def update_memory(self, dataset_train, batch_size=256):
        """
        Update exemplar memory using efficient batched feature extraction
        and herding (aligned with original iCaRL formulation).

        Assumes dataset_train.__getitem__ returns:
            (global_idx, x, y)
        """
        self.model.eval()

        if len(self.seen_classes) == 0:
            return

        # Budget per class (iCaRL standard)
        m = self.memory_size // len(self.seen_classes)

        # 1) Reduce old exemplars (VERY important in CIL)
        for cls in self.old_classes:
            if cls in self.exemplars:
                self.exemplars[cls] = self.exemplars[cls][:m]

        # 2) Herding for new classes
        for cls in self.new_classes:
            # Get subset indices and labels from dataset view
            subset_global_indices = dataset_train.indices
            subset_labels = dataset_train.labels[subset_global_indices]

            # Local indices inside the view belonging to class cls
            local_idxs = np.where(subset_labels == cls)[0]
            if len(local_idxs) == 0:
                continue

            # ---- FAST: use Subset + DataLoader (no per-sample Python loop) ----
            class_subset = Subset(dataset_train, local_idxs)
            loader = DataLoader(
                class_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,  # safer on macOS / older machines
                pin_memory=False
            )

            all_features = []
            global_idxs_for_class = []

            with torch.no_grad():
                for real_idx, x, _ in loader:
                    x = x.to(self.device, non_blocking=True)

                    # Expecting model to return (logits, features)
                    outputs = self.model(x)
                    if isinstance(outputs, tuple):
                        _, feats = outputs
                    else:
                        # fallback: assume feature_extractor exists
                        feats = self.model.feature_extractor(x)

                    feats = F.normalize(feats, dim=1)

                    all_features.append(feats.cpu())
                    global_idxs_for_class.extend(real_idx.numpy())

            # Concatenate once (memory efficient)
            features = torch.cat(all_features, dim=0)  # (N, D)

            # ---- iCaRL ORIGINAL HERDING (more faithful than residual trick) ----
            class_mean = features.mean(dim=0)
            class_mean = F.normalize(class_mean, dim=0)

            selected_exemplars = []
            selected_features_sum = torch.zeros_like(class_mean)

            k_max = min(m, features.shape[0])

            for k in range(k_max):
                # Target: mu * (k+1) - sum(selected)
                target = class_mean * (k + 1) - selected_features_sum

                # Vectorized distance (no clone, no CPU waste)
                distances = torch.norm(features - target, dim=1)

                # Avoid reselection (faster than adding 1e6 noise)
                if selected_exemplars:
                    distances[selected_exemplars] = float("inf")

                chosen = torch.argmin(distances).item()

                selected_exemplars.append(chosen)
                selected_features_sum += features[chosen]

            # Store GLOBAL indices
            chosen_global_indices = np.array(
                [global_idxs_for_class[i] for i in selected_exemplars],
                dtype=np.int64
            )

            self.exemplars[cls] = chosen_global_indices

    def compute_class_means(self, dataset_train, batch_size=64):
        """
        Means are computed from normalized features of exemplars only.
        """
        self.model.eval()
        self.class_means = {}

        if len(self.exemplars) == 0:
            return

        with torch.no_grad():
            for cls in self.seen_classes:
                # Puede pasar en primeras tasks
                if cls not in self.exemplars or len(self.exemplars[cls]) == 0:
                    continue

                exemplar_idxs = self.exemplars[cls]
                feats_list = []

                # Extraer features en batches (más eficiente)
                for i in range(0, len(exemplar_idxs), batch_size):
                    batch_idxs = exemplar_idxs[i:i + batch_size]

                    batch_x = []
                    for idx in batch_idxs:
                        _, x, _ = dataset_train[idx]
                        batch_x.append(x)

                    batch_x = torch.stack(batch_x).to(self.device)

                    # Forward -> embeddings
                    _, feats = self.model(batch_x)

                    # Normalization Features
                    feats = F.normalize(feats, dim=1)

                    feats_list.append(feats.cpu())

                # Concatenate all features list
                feats = torch.cat(feats_list, dim=0)

                # Mean
                mean = feats.mean(dim=0)

                # Normalization (NCM)
                mean = F.normalize(mean, dim=0)

                self.class_means[cls] = mean


    def predict(self, x):
        """
        NCM prediction (iCaRL) using class means.
        x: Tensor shape (D,) or (B, D)
        """
        self.model.eval()

        if len(self.class_means) == 0:
            raise ValueError("class_means is empty. Call compute_class_means() first.")

        with torch.no_grad():
            # Ensure batch dimension
            if x.dim() == 1:
                x = x.unsqueeze(0)

            x = x.to(self.device)

            # Get embeddings
            _, feats = self.model(x)
            feats = F.normalize(feats, dim=1)

            # deterministic class order
            class_ids = sorted(self.class_means.keys())

            means = torch.stack(
                [self.class_means[c].to(self.device) for c in class_ids],
                dim=0
            )  # (C, D)

            # Vectorized NCM distance
            dists = torch.cdist(feats, means, p=2)  # (B, C)
            pred_indices = torch.argmin(dists, dim=1)

            # Map indices -> real class ids (and move to CPU)
            preds = torch.tensor(
                [class_ids[i] for i in pred_indices.cpu().tolist()],
                dtype=torch.long
            )

        return preds

    def evaluate(self, testset, batch_size):
        """
        testset: IDSIncrementalDataset
        """
        self.model.eval()

        y_pred = []
        y_true = []

        loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for real_idx, x, y in loader:
                preds = self.predict(x)

                y_pred.extend(preds.cpu().numpy())
                y_true.extend(y.cpu().numpy())

        return np.array(y_true), np.array(y_pred)

def build_task_icarl2(attack_pattern, attack_classes, classes_names, benign_class=0):

    if sum(attack_pattern) != len(attack_classes):
        raise ValueError("Attack pattern is inconsistence with total number of attacks")

    current_index = 0
    new_attacks = []
    tasks_labels = []
    attack_task_labels = []
    id_to_class = {v: k for k, v in classes_names.items()}

    # Loop start
    for i, n_new in enumerate(attack_pattern):
        if i == 0:
            # Task 1: include benign + first attack chunk
            new_attacks = attack_classes[:n_new]
            current_index += n_new
            attack_task_labels.append(new_attacks)
            tasks_labels.append([benign_class] + new_attacks)
        else:
            new_attacks = attack_classes[current_index:current_index+n_new]
            current_index += n_new
            attack_task_labels.append(new_attacks)
            tasks_labels.append(new_attacks)

    # Convertir a etiquetas numéricas usando el diccionario
    tasks_names = [[id_to_class[i] for i in task] for task in tasks_labels]

    return tasks_labels, tasks_names

def train_icarl(strategy_name, dataset_path, dataset_name, ouput_path, feature_dim, memory_size, T, epochs, batch_size, lr, lambda_kd, attack_pattern, **kwargs):

    datatrain = CILDataset(dataset_path, dataset_name, 'train')
    datatest = CILDataset(dataset_path, dataset_name, 'test')
    benign_class = 0
    attack_classes = list(datatrain.label_to_id.values())[1:]
    input_dim = datatrain.all_features.shape[1]

    #default values
    T = kwargs.get("T", 2.0)
    lambda_kd = kwargs.get("lambda_kd", 5.0)
    
    model = IDSNet(input_dim=input_dim, feature_dim=feature_dim)
    device = get_device()
    icarl = ICaRL(model, device, memory_size)

    tasks_labels, attack_task_labels = build_task_icarl(attack_pattern=attack_pattern, attack_classes=attack_classes, benign_class=benign_class)

    # empy metrics
    num_tasks = len(attack_pattern)
    accuracy_matrix = np.zeros((num_tasks, num_tasks))
    accuracy_attack_matrix =  np.zeros((num_tasks, num_tasks))

    for i in range(len(tasks_labels)):

        print(f'Task {i+1}/ new attacks: {tasks_labels[i]}')

        new_attacks = tasks_labels[i]

        # Dataset for task
        train_dataset_bce, train_dataset_kd = datatrain.build_task_dataset(new_attacks)

        #updates clasiffier
        icarl.add_classes(new_attacks)

        # training
        icarl.train_task(train_dataset_bce,  train_dataset_kd, epochs=epochs, batch_size=batch_size, lr=lr, lambda_kd=lambda_kd, T=T)

        print('finished training')

        # after training
        print('icarl update memory')
        icarl.update_memory(train_dataset_bce, batch_size=batch_size)
        memory_indices = np.concatenate(list(icarl.exemplars.values()))
        print('dataset update memory')
        datatrain.update_memory(memory_indices)

        print('compute class means')
        icarl.compute_class_means(datatrain)

        # Evaluating each task on test set

        stats_test_set = {
            'min_up_to': datatrain.min_up_to,
            'max_up_to': datatrain.max_up_to
        }

        print('compute accuracy tasks')
        test_datasets = datatest.build_task_testset(task_labels=tasks_labels[:i+1],stats=stats_test_set)

        for j in range(len(test_datasets)):
            y_true, y_pred = icarl.evaluate(test_datasets[j], batch_size)
            accuracy_matrix[i, j] = accuracy(y_true, y_pred)
            accuracy_attack_matrix[i, j] = attack_accuracy(y_true, y_pred)

    # compute global metrics
    print(f'accuracy_matrix: {accuracy_matrix}')
    print(f'accuracy_attack_matrix: {accuracy_attack_matrix}')
    print('computing global metrics')
    avg_acc = average_accuracy(accuracy_matrix)
    avg_attack_acc = average_accuracy(accuracy_attack_matrix)
    forgetting_measure = compute_forgetting(accuracy_matrix)
    forgetting_attack = compute_forgetting(accuracy_attack_matrix)

    print(f'average_accuracy: {avg_acc}')
    print(f'average_attack_accuracy: {avg_attack_acc}')
    print(f'average_forgetting: {forgetting_measure}')
    print(f'average_attack_forgetting: {forgetting_attack}')

    save_training_results(dataset_name, strategy_name, attack_pattern, accuracy_matrix, accuracy_attack_matrix, avg_acc, avg_attack_acc, forgetting_measure, forgetting_attack, 'A', ouput_path)