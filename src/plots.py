import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict


def scenario_to_str(pattern):
    """Converts [1,1,1] -> '1-1-1'"""
    return "+".join(map(str, pattern))


def load_results_from_folder(folder_path, dataset_filter=None):

    folder = Path(folder_path)
    results = []

    for json_file in folder.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)

        if dataset_filter is not None and str(data["dataset"]) != str(dataset_filter):
            continue

        results.append(data)

    return results


def plot_metrics_by_scenario(
    folder_path,
    dataset,
    scenario_order,
    save_dir=None,
    figsize=(6, 4),
    dpi=300
):
    """
    Generate 4 line plots:
    - average_accuracy
    - average_attack_accuracy
    - forgetting_measure
    - forgetting_attack_measure
    """

    results = load_results_from_folder(folder_path, dataset_filter=dataset)

    if len(results) == 0:
        raise ValueError(f"No se encontraron JSON para el dataset {dataset}")

    # Fixed Colors
    strategy_colors = {
        "er": "#1f77b4",      # blue
        "icarl": "#2ca02c",   # green
        "der++": "#ff7f0e",     # orange
    }

    display_names = {
        "er": "ER",
        "icarl": "Icarl",
        "der++": "DER++"
    }

    metrics_list = [
        "average_accuracy",
        "average_attack_accuracy",
        "forgetting_measure",
        "forgetting_attack_measure",
    ]

    data_dict = {
        metric: defaultdict(dict) for metric in metrics_list
    }

    for res in results:
        strategy = res["strategy_name"]
        scenario_str = scenario_to_str(res["scenario_pattern"])
        metrics = res["metrics"]

        for metric in metrics_list:
            if metric in metrics:
                data_dict[metric][strategy][scenario_str] = metrics[metric]

    dataset_name = "UNSW-NB15" if str(dataset) == "2015" else "CIC-IDS-2017"

    for metric in metrics_list:
        plt.figure(figsize=figsize, dpi=dpi)

        strategies = sorted(data_dict[metric].keys())

        for strategy in strategies:
            scenario_values = data_dict[metric][strategy]

            y_values = []
            x_labels = []

            for scen in scenario_order:
                scen_str = scen if isinstance(scen, str) else scenario_to_str(scen)

                if scen_str in scenario_values:
                    y_values.append(scenario_values[scen_str])
                else:
                    y_values.append(None)  # gap si falta experimento

                x_labels.append(scen_str)

            strategy_key = strategy.strip().lower()

            color = strategy_colors.get(strategy_key, "#333333")

            label_name = display_names.get(strategy_key, strategy)

            plt.plot(
                x_labels,
                y_values,
                marker="o",
                linewidth=2,
                markersize=5,
                label=label_name,
                color=color
            )

        pretty_metric = metric.replace("_", " ").title()
        plt.title(f"{pretty_metric} - {dataset_name}")
        plt.xlabel("Scenario Pattern")
        plt.ylabel(pretty_metric)
        plt.xticks(rotation=0)
        plt.grid(False)
        plt.legend(frameon=False)

        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                save_path / f"{metric}_dataset_{dataset}.png",
                bbox_inches="tight",
                dpi=dpi
            )

        plt.show()
        plt.close()