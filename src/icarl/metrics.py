from sklearn.metrics import accuracy_score
import json
from pathlib import Path

def accuracy(y_true, y_pred):

  return accuracy_score(y_true, y_pred)

def attack_accuracy(y_true, y_pred):
  mask = y_true != 0

  return accuracy_score(y_true[mask], y_pred[mask])

def average_accuracy(accuracy_matrix):

  """
  accuracy_matrix: list of accuracies up to current task
  """
  num_tasks = accuracy_matrix.shape[0]
  return np.mean([accuracy_matrix[num_tasks-1, k] for k in range(num_tasks)])

import numpy as np

def compute_forgetting(results):
    """
    results: accuracy matrix. results[i, j] = accuracy on task j after training task i
    """
    R = np.array(results)
    T = R.shape[0]

    FM_per_k = []

    for k in range(1, T):  # current task k
        f_jk = []

        for j in range(k):  # all past tasks j < k
            best_past = np.max(R[:k, j])  # max_i a_{i,j}
            current = R[k, j]             # a_{k,j}
            f_jk.append(best_past - current)

        FM_k = np.mean(f_jk)  # (1/(k)) in 0-index = (1/(k-1)) in paper notation
        FM_per_k.append(FM_k)

    return np.mean(FM_per_k)

def save_training_results(dataset_name, strategy_name, attack_pattern, accuracy_matrix, accuracy_attack_matrix, avg_acc, avg_attack_acc, forgetting_measure,forgetting_attack, json_path):

    results_dir = Path(json_path)  # nombre de la carpeta
    results_dir.mkdir(parents=True, exist_ok=True)  # crea la carpeta si no existe

    pattern_str = "-".join(map(str, attack_pattern))

    output_name = f'{dataset_name}_{strategy_name}_{pattern_str}.json'

    file_path = results_dir / output_name

    results = {
            "dataset": dataset_name,
            "strategy_name": strategy_name,
            "scenario_pattern": attack_pattern,
            "metrics": {
                "accuracy_matrix": accuracy_matrix.tolist(),
                "accuracy_attack_matrix": accuracy_attack_matrix.tolist(),
                "average_accuracy": avg_acc,
                "average_attack_accuracy": avg_attack_acc,
                "forgetting_measure": forgetting_measure,
                "forgetting_attack_measure": forgetting_attack
            }
        }

    with open(file_path, "w") as f:
        json.dump(results, f, indent=2)