import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from src.utils import print_task_results
from src.metrics import accuracy, macro_f1, compute_cm, save_confusion_matrix, average_accuracy
from src.model import CILModel
from src.task_builder import UpToNormalizer, build_scenario, build_task

def train_one_task(model, train_loader, optimizer, device, num_epochs, class_weights=None):

    model.train()

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.long().to(device)

            optimizer.zero_grad()

            logits, _ = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

#TODO check accuracy and accuracy_attack matrix respect Margarita's indications
def evaluate(model, dataset, seen_classes, task_id, device):
    model.eval()

    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).long()

            logits, _ = model(x)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Overall accuracy
    acc = accuracy(all_labels, all_preds)

    # Macro F1
    f1 = macro_f1(all_labels, all_preds)

    # Attack accuracy (exclude Normal = class 0)
    attack_indices = [i for i, label in enumerate(all_labels) if label != 0]

    if len(attack_indices) > 0:
        attack_labels = [all_labels[i] for i in attack_indices]
        attack_preds = [all_preds[i] for i in attack_indices]
        acc_attack = accuracy(attack_labels, attack_preds)
    else:
        acc_attack = 0.0

    # Confusion matrix
    cm = compute_cm(all_labels, all_preds, seen_classes, show_plot=False)
    save_path = f"results/confusion_matrices/task_{task_id}.png"
    save_confusion_matrix(cm, seen_classes, save_path)

    return acc, acc_attack, f1

def update_buffer(buffer_indices, task_dataset, total_buffer_size, seen_classes):

    # Full original dataset
    full_dataset = task_dataset.dataset

    # Current task indices (relative to full dataset)
    current_indices = task_dataset.indices

    # Merge old + new
    combined_indices = list(set(buffer_indices + list(current_indices)))

    num_classes = len(seen_classes)
    memory_per_class = total_buffer_size // num_classes

    new_buffer = []

    for class_id in range(num_classes):

        class_samples = [
            idx for idx in combined_indices
            if full_dataset.y[idx].item() == class_id
        ]

        if len(class_samples) > memory_per_class:
            class_samples = random.sample(class_samples, memory_per_class)

        new_buffer.extend(class_samples)

    return new_buffer

def train_and_evaluate_ER(
    trainset,
    feature_dim,
    device,
    memory_size,
    attack_pattern,
    epochs,
    learning_rate=0.0003,
    weight_decay=1e-4,
    batch_size=512,
    dataset_name="CICIDS",
    **kwargs):

    print("\nStrategy ExperienceReplay ========")
    print(f"\n=== Scenario - {attack_pattern} ===\n")

    # -------------------------
    # Reset model
    # -------------------------
    model = CILModel(
        input_dim=trainset.x.shape[1],
        feature_dim=feature_dim
    ).to(device)

    normalizer = UpToNormalizer()
    buffer_indices = []

    tasks, _ = tasks, pattern = build_scenario(
        all_classes=trainset.classes,
        attacks_pattern=attack_pattern,
        benign_class=trainset.benign
        )
    num_tasks = len(tasks)

    accuracy_matrix = [[0]*num_tasks for _ in range(num_tasks)]
    attack_accuracy_matrix = [[0]*num_tasks for _ in range(num_tasks)]
    f1_history = []

    # =========================
    # FULL ER LOOP
    # =========================
    for task_index, task_classes in enumerate(tasks):

        task_dataset = build_task(trainset, task_classes)

        # Expand classifier
        num_classes = len(task_classes)
        model.update_classifier(num_classes)
        model = model.to(device)

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # -------------------------
        # Normalize new task data
        # -------------------------
        task_x = task_dataset.dataset.x[task_dataset.indices].numpy()
        normalizer.update(task_x)

        normalized_x = normalizer.normalize(task_x)
        task_dataset.dataset.x[task_dataset.indices] = torch.tensor(
            normalized_x,
            dtype=torch.float32
        )

        # -------------------------
        # Merge buffer + new task
        # -------------------------
        combined_indices = buffer_indices + list(task_dataset.indices)
        combined_dataset = Subset(trainset, combined_indices)

        # -------------------------
        # Class weights
        # -------------------------
        labels = trainset.y[combined_indices].numpy()
        class_counts = np.bincount(labels, minlength=num_classes)
        class_weights = 1.0 / (class_counts + 1e-8)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

        # -------------------------
        # Train
        # -------------------------
        train_loader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        train_one_task(
            model,
            train_loader,
            optimizer,
            device,
            epochs,
            class_weights
        )

        # -------------------------
        # Evaluate on seen tasks
        # -------------------------
        for prev_index in range(task_index + 1):

            prev_classes = tasks[prev_index]
            prev_dataset = build_task(trainset, prev_classes)

            acc, acc_attack, f1 = evaluate(
                model,
                prev_dataset,
                prev_classes,
                f"{task_index+1}_eval_on_task_{prev_index+1}",
                device
            )

            accuracy_matrix[task_index][prev_index] = acc
            attack_accuracy_matrix[task_index][prev_index] = acc_attack

            if prev_index == task_index:
                seen_classes = []
                for t in tasks[:task_index+1]:
                    for c in t:
                        if c not in seen_classes:
                            seen_classes.append(c)

                f1_history.append(f1)

                print_task_results(
                    task_index + 1,
                    task_classes,
                    seen_classes,
                    acc,
                    acc_attack,
                    f1
                )

        # -------------------------
        # Update buffer
        # -------------------------
        buffer_indices = update_buffer(
            buffer_indices,
            task_dataset,
            memory_size,
            task_classes
        )

    # =========================
    # FINAL METRICS
    # =========================

    final_accuracies = accuracy_matrix[num_tasks-1]
    final_attack_accuracies = attack_accuracy_matrix[num_tasks-1]

    avg_acc = np.mean(final_accuracies)
    avg_attack_acc = np.mean(final_attack_accuracies)

    forgetting_values = []
    forgetting_attack_values = []

    for t in range(num_tasks - 1):
        max_past = max(accuracy_matrix[i][t] for i in range(num_tasks))
        final_perf = accuracy_matrix[num_tasks-1][t]
        forgetting_values.append(max_past - final_perf)

        max_past_attack = max(attack_accuracy_matrix[i][t] for i in range(num_tasks))
        final_perf_attack = attack_accuracy_matrix[num_tasks-1][t]
        forgetting_attack_values.append(max_past_attack - final_perf_attack)

    forgetting = np.mean(forgetting_values)
    forgetting_attack = np.mean(forgetting_attack_values)

    print("\nFinal Average Accuracy:", avg_acc)
    print("Final Average Attack Accuracy:", avg_attack_acc)
    print("Final Forgetting:", forgetting)
    print("Final Attack Forgetting:", forgetting_attack)

    #TODO modify

    # save_training_results(
    #     dataset_name,
    #     "ExperienceReplay",
    #     attack_pattern,
    #     accuracy_matrix,
    #     attack_accuracy_matrix,
    #     f1_history,
    #     avg_acc,
    #     avg_attack_acc,
    #     forgetting,
    #     forgetting_attack,
    #     scenario_id,
    #     f"cil-intrusion-detection/results/training/ExperienceReplay_{scenario_id}.json"
    # )

    return avg_acc, forgetting
