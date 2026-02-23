import os
import json
import numpy as np
import matplotlib.pyplot as plt

def line_plot(res_path="results/training"):

    # Structure: dataset -> strategy -> metrics
    results = {}

    # Loop over all JSON files
    for file_name in os.listdir(res_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(res_path, file_name)  # FIXED (was 'res')

            with open(file_path, "r") as f:
                data = json.load(f)

            dataset = data.get("dataset", "unknown_dataset")
            strategy = data["strategy_name"]
            avg_acc = data["metrics"]["average_accuracy"]
            forgetting = data["metrics"]["forgetting_measure"]

            # Initialize dataset
            if dataset not in results:
                results[dataset] = {}

            # Initialize strategy inside dataset
            if strategy not in results[dataset]:
                results[dataset][strategy] = {
                    "scenarios": [],
                    "avg_accuracy": [],
                    "forgetting": []
                }

            # Append values
            strategy_data = results[dataset][strategy]
            scenario_id = len(strategy_data["scenarios"]) + 1

            strategy_data["scenarios"].append(scenario_id)
            strategy_data["avg_accuracy"].append(avg_acc)
            strategy_data["forgetting"].append(forgetting)

    # ---------------- PLOTTING ----------------
    for dataset, strategies in results.items():
        for strategy, values in strategies.items():

            plt.figure(figsize=(8, 5))

            plt.plot(values["scenarios"], values["avg_accuracy"],
                     marker="o", label="Average Accuracy")

            plt.plot(values["scenarios"], values["forgetting"],
                     marker="s", label="Forgetting Measure")

            plt.xlabel("Scenario")
            plt.ylabel("Metric Value")
            plt.title(f"{dataset} - {strategy}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

def bar_plot(target_scenario, res_path="results/training"):

    # Structure: dataset -> strategy -> metrics
    results = {}

    # Read all JSON files
    for file_name in os.listdir(res_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(res_path, file_name)

            with open(file_path, "r") as f:
                data = json.load(f)

            dataset = data.get("dataset", "unknown_dataset")
            scenario_id = data.get("scenario_id")

            # Filter only by scenario
            if scenario_id == target_scenario:

                strategy = data["strategy_name"]
                metrics = data["metrics"]

                if dataset not in results:
                    results[dataset] = {}

                results[dataset][strategy] = {
                    "avg_acc": metrics.get("average_accuracy", 0),
                    "avg_attack_acc": metrics.get("average_attack_accuracy", 0),
                    "forgetting": metrics.get("forgetting_measure", 0),
                    "forgetting_attack": metrics.get("forgetting_attack_measure", 0),
                }

    # --------- PLOTTING ------------
    for dataset, strategies in results.items():

        for strategy, values in strategies.items():

            labels = [
                "Avg Accuracy",
                "Avg Attack Accuracy",
                "Forgetting",
                "Forgetting Attack"
            ]

            data_values = [
                values["avg_acc"],
                values["avg_attack_acc"],
                values["forgetting"],
                values["forgetting_attack"]
            ]

            x = np.arange(len(labels))

            plt.figure(figsize=(8, 5))
            bars = plt.bar(x, data_values)

            plt.xticks(x, labels, rotation=20)
            plt.ylabel("Metric Value")
            plt.title(f"{dataset} - {strategy} - Scenario {target_scenario}")
            plt.ylim(0, 1)
            plt.grid(axis='y', linestyle='--', alpha=0.6)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01,
                    f"{height:.3f}",
                    ha='center',
                    va='bottom'
                )

            plt.tight_layout()
            plt.show()