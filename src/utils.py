import json
import argparse
import torch
import numpy as np
import random
import math
from pathlib import Path

def save_training_results(dataset_name, strategy_name, attack_pattern, accuracy_matrix, accuracy_attack_matrix,
    avg_acc, avg_attack_acc, forgetting_measure,forgetting_attack, json_path):

    output_name = f'{dataset_name}_{strategy_name}_{attack_pattern}.json'

    file_path = f"{json_path}/{output_name}"

    results = {
            "dataset": dataset_name,
            "strategy_name": strategy_name,
            "scenario_pattern": attack_pattern,
            "metrics": {
                "accuracy_matrix": accuracy_matrix,
                "accuracy_attack_matrix": accuracy_attack_matrix,
                "average_accuracy": avg_acc,
                "average_attack_accuracy": avg_attack_acc,
                "forgetting_measure": forgetting_measure,
                "forgetting_attack_measure": forgetting_attack
            }
        }

    with open(file_path, "w") as f:
        json.dump(results, f, indent=2)

def print_task_results(task_num, new_attacks, seen_attacks, accuracy, acc_attack, macro_f1):
    """
     --- Task 1 ---
    
    New attacks: [benign, dos]
    Seen so far: [portscan]

    accuracy: 0.00
    macro-f1: 0.00
    """
    accuracy_trunc = math.trunc(accuracy*100)/100 
    acc_attack_trunc = math.trunc(acc_attack*100)/100 
    macro_f1_trunc = math.trunc(macro_f1*100)/100 
    print(f"   --- Task {task_num} ---\n\n",
        f"   New attacks: {new_attacks}\n",
        f"   Seen so far: {seen_attacks}\n\n",
        f"   accuracy: {accuracy_trunc:.2f}\n",
        f"   attack_accuracy: {acc_attack_trunc:.2f}\n",
        f"   macro-f1: {macro_f1_trunc:.2f}\n\n")

def print_scenario(attack_pattern):
    print(f"=== Scenario - {attack_pattern} ===\n\n")

def print_strategy(strategy_name):
    print(f"Strategy {strategy_name} ========\n\n")

def print_final_metrics(forgetting, forgetting_attack, avg_acc, avg_attack_acc):
    forgetting_trunc = math.trunc(forgetting*100)/100
    forgetting_attack_trunc = math.trunc(forgetting_attack*100)/100
    avg_acc_trunc = math.trunc(avg_acc*100)/100 
    avg_attack_trunc = math.trunc(avg_attack_acc*100)/100 
    print(f"Forgetting measure: {forgetting_trunc:.2f}")
    print(f"Forgetting attack measure: {forgetting_attack_trunc:.2f}")
    print(f"Average accuracy: {avg_acc_trunc:.2f}")
    print(f"Average attack accuracy: {avg_attack_trunc:.2f}\n")

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
    parser.add_argument('--strategy', type=str, default='er', choices=['er', 'icarl', 'der'],
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
    parser.add_argument("--batch_size", type=int, default=64,
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



