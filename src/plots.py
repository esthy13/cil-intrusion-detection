import os
import json
import numpy as np
import matplotlib.pyplot as plt

def line_plot(res_path="results/training"):

    results = {}

    # --------- LOAD DATA ----------
    for file_name in os.listdir(res_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(res_path, file_name)

            with open(file_path, "r") as f:
                data = json.load(f)

            dataset = data.get("dataset", "unknown_dataset")
            strategy = data["strategy_name"]
            scenario_pattern = data.get("scenario_pattern", [])

            avg_acc = data["metrics"]["average_accuracy"]
            forgetting = data["metrics"]["forgetting_measure"]

            if dataset not in results:
                results[dataset] = {}

            if strategy not in results[dataset]:
                results[dataset][strategy] = {
                    "patterns": [],
                    "avg_accuracy": [],
                    "forgetting": []
                }

            strategy_data = results[dataset][strategy]

            strategy_data["patterns"].append(scenario_pattern)
            strategy_data["avg_accuracy"].append(avg_acc)
            strategy_data["forgetting"].append(forgetting)

    # --------- PLOTTING ----------
    for dataset, strategies in results.items():
        for strategy, values in strategies.items():

            num_points = len(values["avg_accuracy"])
            x_positions = list(range(1, num_points + 1))  # integers only

            plt.figure(figsize=(10, 5))

            plt.plot(x_positions, values["avg_accuracy"],
                     marker="o", label="Average Accuracy")

            plt.plot(x_positions, values["forgetting"],
                     marker="s", label="Forgetting Measure")

            # Set integer ticks
            plt.xticks(
                x_positions,
                [str(p) for p in values["patterns"]],
                rotation=45,
                ha="right"
            )

            plt.xlabel("Scenario Pattern")
            plt.ylabel("Metric Value")
            plt.title(f"{dataset} - {strategy}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
def bar_plot(target_scenario_pattern, res_path="results/training"):

    results = {}

    for file_name in os.listdir(res_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(res_path, file_name)

            with open(file_path, "r") as f:
                data = json.load(f)

            dataset = data.get("dataset", "unknown_dataset")
            scenario_pattern = tuple(data.get("scenario_pattern", []))

            # Filter using scenario_pattern
            if scenario_pattern == tuple(target_scenario_pattern):

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

    # print("Collected results:", results)

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
            plt.title(f"{dataset} - {strategy}")
            plt.ylim(0, 1)
            plt.grid(axis='y', linestyle='--', alpha=0.6)

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