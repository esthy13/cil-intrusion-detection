import json
import argparse
import torch
import numpy as np
import random
import math

def print_strategy_pattern(strategy_name, attack_pattern):
    print(f"\n=== Strategy {strategy_name} - attack pattern {attack_pattern} ===\n")

def print_final_metrics(accuracy_matrix, accuracy_attack_matrix, avg_acc, avg_attack_acc, forgetting_measure, forgetting_attack):
    print("\naccuracy_matrix:\n", np.round(accuracy_matrix, 4))
    print(f"\naccuracy_attack_matrix\n", np.round(accuracy_attack_matrix, 4))
    print(f"\navg_acc: {avg_acc:.4f}")
    print(f"\navg_attack_acc: {avg_attack_acc:.4f}")
    print(f"\nforgetting: {forgetting_measure:.4f}")
    print(f"\nforgetting_attack: {forgetting_attack:.4f}\n")

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def build_parser():
    parser = argparse.ArgumentParser(description='Continous incremental learning training')
    parser.register('type', None, str.lower)
    parser.add_argument('--strategy', type=str, default='icarl', choices=['er', 'icarl', 'der'],
                        help='CIL strategy to use (e.g., er, icarl, der)')
    parser.add_argument('--dataset', type=int, default=2017, choices=[2015, 2017],
                        help='Dataset to use')
    parser.add_argument("--scenarios",
    nargs="+",
    type=str,
    required=True,
    help='Example: --scenarios "1+1+1" "2+2+2"')
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of epochs to train per scenario")
    parser.add_argument("--lr", type=float, default=0.01,
                        help = "learning rate")
    parser.add_argument("--memory_size", type=int, default=7000,
                        help = "Total memory size for old classes")
    parser.add_argument("--feature_dim", type=int, default=128,
                        help = "feature embeddings output")
    parser.add_argument("--batch_size", type=int, default=128,
                        help = "batch size")

    args, unknown = parser.parse_known_args()
    
    args.scenarios = parse_scenarios(args.scenarios)
    strategy_kwargs = parse_unknown_kwargs(unknown)

    return args, strategy_kwargs

def parse_unknown_kwargs(unknown):
    """
    Converts unknown CLI arguments into a kwargs-style dictionary.
    Example:
    ["--alpha", "0.5", "--beta", "1"] -> {"alpha": 0.5, "beta": 1}
    """
    kwargs = {}
    i = 0

    while i < len(unknown):
        key = unknown[i]

        if key.startswith("--"):
            key = key.lstrip("-").replace("-", "_")

            # Si el siguiente elemento NO es otro flag, es su valor
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                value = unknown[i + 1]
                i += 1
            else:
                # Flag tipo booleano (--debug)
                value = True

            # Intentar casteo automático (int, float, bool)
            if isinstance(value, str):
                if value.lower() in ["true", "false"]:
                    value = value.lower() == "true"
                else:
                    try:
                        if "." in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass  # se queda como string

            kwargs[key] = value

        i += 1

    return kwargs

def parse_scenarios(raw_scenarios):
    """
    Converts ["1+1+1", "2+2+2"] -> [[1,1,1], [2,2,2]]
    """
    parsed = []
    for s in raw_scenarios:
        try:
            parsed.append([int(x) for x in s.split("+")])
        except ValueError:
            raise ValueError(
                f"Invalid scenario format: '{s}'. Use format like 1+1+1"
            )
    return parsed

def build_task_icarl(attack_pattern, attack_classes, classes_names, benign_class=0):

    if sum(attack_pattern) != len(attack_classes):
        raise ValueError("Attack pattern is inconsistence with total number of attacks")

    current_index = 0
    new_attacks = []
    tasks_labels = []
    attack_task_labels = []
    id_to_class = {v: k for k, v in classes_names.items()}

    # Loop
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

    # Covert numeric labels to classes
    tasks_names = [[id_to_class[i] for i in task] for task in tasks_labels]

    return tasks_labels, tasks_names



