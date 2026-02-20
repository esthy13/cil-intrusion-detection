import os
import torch
import numpy as np
from src.task_builder import build_scenario, UpToNormalizer, build_task
from src.model import CILModel
from src.der import ReservoirBuffer, train_task, evaluate
from src.utils import print_task_results, save_training_results
from src.metrics import compute_forgetting, average_accuracy
from torch.utils.data import DataLoader

def train_and_evaluate_DER(
    scenario_id,
    trainset,
    testset,
    feature_dim,
    device,
    memory_size,
    attack_pattern, # just one pattern, single array
    epochs
    ):

    input_dim = trainset.x.shape[1]

    # get all classes from dataset
    all_classes = trainset.classes

    os.makedirs("results/confusion_matrices", exist_ok=True)
    os.makedirs("results/training", exist_ok=True)

    # build scenario from attack pattern
    tasks, attack_pattern = build_scenario(all_classes, attack_pattern, benign_class=trainset.benign)

    # Initialize model and buffer
    model = CILModel(input_dim).to(device)
    reservoir_buffer = ReservoirBuffer(size=memory_size)

    acc_history = []
    f1_history = []
    seen_classes = []

    # Initialize a matrix to store accuracy values a_{k,j}
    # This will have dimensions (number_of_tasks, number_of_tasks)
    # e.g., if there are 3 tasks: a_matrix = [[a_0, a_1, a_2], [a_1, a_1, a_2], [a_2, a_2, a_2]]
    a_matrix = np.zeros((len(tasks), len(tasks)))

    for task_id, classes in enumerate(tasks):

        train_norm = trainset.copy()
        test_norm = testset.copy()

        # update classifier and optimizer
        model.update_classifier(len(classes))
        model.classifier = model.classifier.to(device)

        #TODO modify optimizer like margarita told you to do it
        lr = 1e-5
        optimizer = torch.optim.AdamW( model.parameters(), lr=lr ) 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, 
        T_max=epochs, eta_min=lr * 0.01 )
        normalizer = UpToNormalizer()

        # up-to-normalization
        task_dataset = build_task(trainset, classes)
        task_x = np.stack([task_dataset[i][0].numpy() for i in range(len(task_dataset))])
        normalizer.update(task_x)

        train_norm.set_features(normalizer.normalize(trainset.x))
        test_norm.set_features(normalizer.normalize(testset.x))

        train_loader = DataLoader(
            build_task(train_norm, classes),
            batch_size=128,
            shuffle=True
        )

        # train model on the current task
        train_task(model, train_loader, reservoir_buffer, optimizer, scheduler, device, epochs)

        # good I am evaluating on the test set!!!
        # evaluate the model on the current task
        acc, f1, y_true, y_pred = evaluate(model, test_norm, classes, device)

        # update the history
        acc_history.append(acc)
        f1_history.append(f1)

        # update the accuracy matrix with the accuracy for all previous tasks
        for prev_task_id in range(task_id + 1):  # evaluate the current model on all previous tasks
            prev_seen_classes = tasks[prev_task_id]
            prev_acc, _, _, _ = evaluate(model, testset, prev_seen_classes, device)
            a_matrix[task_id, prev_task_id] = prev_acc

        # new attacks = classes - seen_classes
        new_classes = [c for c in classes if c not in seen_classes]

        print_task_results(task_id+1, new_classes, seen_classes, acc, f1)

        seen_classes += new_classes

    # compute metrics for this scenario
    forgetting_measure = compute_forgetting(a_matrix)
    avg_acc = average_accuracy(acc_history)

    # save results for this scenario
    out_json = f"results/training/DER_scenario_{scenario_id+1}_results.json"
    save_training_results("DER++", attack_pattern, acc_history, f1_history,
    avg_acc, forgetting_measure, scenario_id, out_json)

    # TODO save the weights for each scenario and upload them somewhere (maybe github) if they don't weight too much
