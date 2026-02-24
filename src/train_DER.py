import os
import torch
import numpy as np
from src.icarl.utils import get_device 
from src.task_builder import build_scenario, UpToNormalizer, build_task
from src.model import CILModel
from src.der import ReservoirBuffer, train_task, evaluate
from src.utils import print_task_results, save_training_results, print_final_metrics, print_strategy, print_scenario, unzip_if_needed
from src.metrics import compute_forgetting, average_accuracy
from torch.utils.data import DataLoader
from src.dataset import IDSBaseDataset

def train_all_scenarios_der(
    strategy_name,
    dataset_path,
    dataset_name,
    output_path,
    feature_dim,
    memory_size,
    epochs,
    batch_size,
    lr,
    attack_patterns,
    ):

    device = get_device()

    DATA_ROOT = f"{dataset_path}/{dataset_name}"

    if dataset_name == 2015:
        unzip_if_needed(dataset_path, 2015)
        train_set = IDSBaseDataset(DATA_ROOT, split="train", target_col="attack_cat", benign_class="Normal")
        test_set  = IDSBaseDataset(DATA_ROOT, split="test", target_col="attack_cat", benign_class= "Normal")

    elif dataset_name == 2017:
        unzip_if_needed(dataset_path, 2017)
        train_set = IDSBaseDataset(DATA_ROOT, split="train")
        test_set  = IDSBaseDataset(DATA_ROOT, split="test")

    else:
        print("Dataset not supported yet")
        return

    print_strategy(strategy_name)

    for scenario_id, attack_pattern in enumerate(attack_patterns):
        print_scenario(attack_pattern)
        train_and_evaluate_DER(
            scenario_id,
            train_set,
            test_set,
            feature_dim,
            device,
            memory_size,
            attack_pattern, # single pattern
            epochs,
            output_path,
            batch_size,
            lr
        )

def train_and_evaluate_DER(
    scenario_id,
    trainset,
    testset,
    feature_dim,
    device,
    memory_size,
    attack_pattern, # just one pattern, single array
    epochs,
    output_path,
    batch_size=128,
    lr = 1e-3
    ):

    input_dim = trainset.x.shape[1]

    # get all classes from dataset
    all_classes = trainset.classes

    os.makedirs("results/confusion_matrices", exist_ok=True)
    os.makedirs("results/training", exist_ok=True)
    os.makedirs("results/weights", exist_ok=True)

    # build scenario from attack pattern
    tasks, attack_pattern = build_scenario(all_classes, attack_pattern, benign_class=trainset.benign)

    # Initialize model and buffer
    model = CILModel(input_dim, feature_dim).to(device)
    reservoir_buffer = ReservoirBuffer(size=memory_size)

    acc_history = []
    acc_attack_history = []
    seen_classes = []

    # Initialize a matrix to store accuracy values a_{k,j}
    # This will have dimensions (number_of_tasks, number_of_tasks)
    # e.g., if there are 3 tasks: a_matrix = [[a_0, a_1, a_2], [a_1, a_1, a_2], [a_2, a_2, a_2]]
    a_matrix = np.zeros((len(tasks), len(tasks)))
    attack_accuracy = np.zeros((len(tasks), len(tasks)))

    for task_id, task_classes in enumerate(tasks):

        train_norm = trainset.copy()
        test_norm = testset.copy()

        # update classifier and optimizer
        model.update_classifier(len(task_classes))
        model.classifier = model.classifier.to(device)

        lr = lr
        optimizer = torch.optim.Adam(model.parameters(), lr=lr ) 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=lr * 0.01
        )

        normalizer = UpToNormalizer()

        # up-to-normalization
        task_dataset = build_task(trainset, task_classes)
        task_x = np.stack([task_dataset[i][0].numpy() for i in range(len(task_dataset))])
        
        normalizer.update(task_x)

        train_norm.set_features(normalizer.normalize(trainset.x))
        test_norm.set_features(normalizer.normalize(testset.x))

        train_loader = DataLoader(
            build_task(train_norm, task_classes),
            batch_size=batch_size,
            shuffle=True
        )

        # train model on the current task
        train_task(model, train_loader, reservoir_buffer, optimizer, scheduler, device=device, epochs= epochs)

        # evaluate the model on the current task
        acc, acc_attack, _, _ = evaluate(model, test_norm, task_classes, device, benign_class=trainset.benign)

        # update the history
        acc_history.append(acc)
        acc_attack_history.append(acc_attack)

        # update the accuracy matrix with the accuracy for all previous tasks
        for prev_task_id in range(task_id + 1):  # evaluate the current model on all previous tasks
            prev_seen_classes = tasks[prev_task_id]

            # Evaluate once using the new function
            prev_acc, attack_acc, _ , _ = evaluate(
                model, 
                test_norm, 
                prev_seen_classes, 
                device, 
                benign_class=trainset.benign
            )

            a_matrix[task_id, prev_task_id] = prev_acc
            attack_accuracy[task_id, prev_task_id] = attack_acc

        # new attacks = classes - seen_classes
        new_classes = [c for c in task_classes if c not in seen_classes]

        print_task_results(task_id+1, new_classes, seen_classes, acc, acc_attack)

        seen_classes += new_classes

        # saving the model weights here
        weights_path = f"results/weights/DER_scenario{scenario_id}_task{task_id}.weights.h5"
        model.save_weights(weights_path)

    # compute metrics for this scenario
    forgetting_measure = compute_forgetting(a_matrix)
    avg_acc = average_accuracy(acc_history)
    avg_attack = average_accuracy(acc_attack_history)
    forgetting_attack = compute_forgetting(attack_accuracy)

    # save results for this scenario
    save_training_results(trainset.name, "DER++", attack_pattern, a_matrix.tolist(), attack_accuracy.tolist(),
    avg_acc, avg_attack, forgetting_measure,forgetting_attack, output_path)
    print_final_metrics(forgetting_measure, forgetting_attack, avg_acc, avg_attack)
