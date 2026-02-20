import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import cycle
from torch.utils.data import DataLoader
from tqdm import tqdm

#Import functions
from src.model import IDSNet
from src.metrics import accuracy, average_accuracy, compute_forgetting, save_confusion_matrix, compute_cm
from src.utils import save_training_results, print_task_results, print_scenario, print_strategy, get_device

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
                loss_ce = F.cross_entropy(student_logits, y_bce)

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

    def update_memory(self, dataset_train):

        self.model.eval()

        m = self.memory_size // len(self.seen_classes)

        # 1. Reduce old exemplars
        for cls in self.old_classes:
            if cls in self.exemplars:
                self.exemplars[cls] = self.exemplars[cls][:m]

        # 2. Construct exemplars for new classes using HERDING
        for cls in self.new_classes:
            idxs = np.where(dataset_train.labels == cls)[0]

            # Extract features for all samples of the class
            features = []
            with torch.no_grad():
                for idx in idxs:
                    _, x, _ = dataset_train[idx]
                    x = x.unsqueeze(0).to(self.device)
                    _, f = self.model(x)
                    f = F.normalize(f, dim=1)
                    features.append(f.cpu())

            features = torch.cat(features, dim=0)  # (N, D)

            # True class mean (normalized features)
            class_mean = features.mean(dim=0)
            class_mean = F.normalize(class_mean, dim=0)

            # Herding selection
            selected_exemplars = []
            selected_features = []

            for k in range(min(m, len(idxs))):
                if k == 0:
                    # First exemplar: closest to class mean
                    distances = torch.norm(features - class_mean, dim=1)
                    chosen = torch.argmin(distances).item()
                else:
                    # Compute mean of already selected exemplars
                    current_mean = torch.stack(selected_features, dim=0).mean(dim=0)
                    current_mean = F.normalize(current_mean, dim=0)

                    # Select sample that best reduces mean error
                    residual = class_mean - current_mean
                    distances = torch.norm(features - residual, dim=1)

                    chosen = torch.argmin(distances).item()

                selected_exemplars.append(idxs[chosen])
                selected_features.append(features[chosen])

                # Prevent reselection
                features[chosen] = features[chosen] + 1e6

            self.exemplars[cls] = np.array(selected_exemplars)

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

                    # Forward -> obtener embeddings
                    _, feats = self.model(batch_x)

                    # Normalización L2 (CRÍTICA en iCaRL)
                    feats = F.normalize(feats, dim=1)

                    feats_list.append(feats.cpu())

                # Concatenar todas las features de los exemplars
                feats = torch.cat(feats_list, dim=0)

                # Media de features normalizadas
                mean = feats.mean(dim=0)

                # Normalizar el prototipo final (NCM)
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

def train_icarl(root_dir_dataset, dataset_name, out_path, json_path, feature_cols, label_col, feature_dim, memory_size, T, epochs, batch_size, lr, lambda_kd, attack_pattern):

    if sum(attack_pattern) != trainset.total_labels:
      raise ValueError("Attacks pattern is inconsistence with total number of attacks")

    datatrain = CILDataset(root_dir_dataset, dataset_name, 'train', label_col, feature_cols)
    datatest = CILDataset(root_dir_dataset, dataset_name, 'test', label_col, feature_cols)
    benign_class = 0
    attack_classes = list(datatrain.label_to_id.values())[1:]
    input_dim = datatrain.all_features.shape[1]

    model = IDSNet(input_dim=input_dim, feature_dim=feature_dim)
    device = get_device()

    icarl = ICaRL(model, device, memory_size)

    # empy metrics
    num_tasks = len(attack_pattern)
    acc_history = []
    f1_history = []
    results_acc_per_task = np.zeros((num_tasks, num_tasks))
    results_f1_per_task =  np.zeros((num_tasks, num_tasks))
    test_datasets = []

    current_index = 0
    new_attacks = None
    seen_so_far = None

    # Loop start
    for i, n_new in enumerate(attacks_pattern):
        #generate scenario
        if i == 0:
            # Task 1: include benign + first attack chunk
            new_attacks = [benign_class] + attack_classes[:n_new]
            current_index += n_new
            seen_so_far = new_attacks

        else:
            new_attacks = attack_classes[current_index:current_index+n_new]
            current_index += n_new
            seen_so_far = [benign_class] + attack_classes[:current_index]

        print(f"\nTask {i+1}")
        print(f"New attacks: {new_attacks}")

        # Dataset for task
        train_dataset_bce, train_dataset_kd = datatrain.build_task_dataset(new_attacks)

        #updates clasiffier
        icarl.add_classes(new_attacks)
        
        # training
        icarl.train_task(train_dataset_bce,  train_dataset_kd, epochs=epochs, batch_size=batch_size, lr=lr, lambda_kd=lambda_kd, T=T)

        # after training
        icarl.update_memory(train_dataset_bce)
        datatrain.update_memory(icarl.exemplars)
        icarl.compute_class_means(datatrain)

        # evaluate test set
        datatest.min_up_to, datatest.max_up_to = datatrain.min_up_to, datatrain.max_up_to

        # compute statistics per task
        testset, _ = datatest.build_task_dataset(new_attacks)
        test_datasets.append(testset)
        
        for j in range(len(test_datasets)):
            y_true, y_pred = icarl.evaluate(testset, batch_size)
            results_acc_per_task[i, j] = accuracy(y_true, y_pred)
            results_f1_per_task[i, j] = macro_f1(y_true, y_pred)

        # compute statistics on all tasks
        testset2, _ = datatest.build_task_dataset([seen_so_far])
        y_true2, y_pred2 = icarl.evaluate(testset2, batch_size)
        acc_history.append(accuracy(y_true2, y_pred2))
        f1_history.append(macro_f1(y_true2, y_pred2, average='macro'))
        cm = compute_cm(y_true2, y_pred2, seen_so_far, show_plot=False)
        save_confusion_matrix(cm, seen_so_far, out_path, title=None)

    #forgetting
    forgetting_measure = compute_forgetting(results_acc_per_task)
    avg_acc = average_accuracy(acc_history)

    # export json
    save_training_results(
        strategy_name='icarl',
        attack_pattern=attack_pattern,
        acc_history=acc_history,
        f1_history=f1_history,
        avg_acc=avg_acc,
        forgetting_measure=forgetting_measure,
        scenario_id="scenario_id",
        json_path=json_path)












