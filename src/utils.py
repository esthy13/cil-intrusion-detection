import json

def save_training_results(strategy_name, attack_pattern, acc_history, f1_history,
    avg_acc, forgetting_measure, scenario_id):
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

    out_json = f"results/training/DER_scenario_{scenario_id+1}_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)