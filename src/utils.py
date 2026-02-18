import json

def save_training_results(strategy_name, attack_pattern, acc_history, f1_history,
    avg_acc, forgetting_measure, scenario_id, json_path):
    results = {
            "strategy_name": strategy_name,
            "scenario_pattern": attack_pattern,
            "metrics": {
                "accuracy": acc_history,
                "f1": f1_history,
                "average_accuracy": avg_acc,
                "forgetting_measure": forgetting_measure
            }
        }

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

def print_task_results(task_num, new_attacks, seen_attacks, accuracy, macro_f1):
    """
     --- Task 1 ---
    
    New attacks: [benign, dos]
    Seen so far: [portscan]

    accuracy: 0.00
    macro-f1: 0.00
    """
    print(f"   --- Task {task_num} ---\n\n",
        f"   New attacks: {new_attacks}\n",
        f"   Seen so far: {seen_attacks}\n\n",
        f"   accuracy: {accuracy:.2f}\n",
        f"   macro-f1: {macro_f1:.2f}\n\n")