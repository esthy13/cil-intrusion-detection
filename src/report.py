import json
from pathlib import Path
from collections import defaultdict
import math

def generate_cil_metric_table(json_folder, dataset_filter=None):
    """
    Generate LaTeX CIL table with:
    - Dataset filtering
    - Bold headers
    - Best values highlighted:
        * MAX for AA, AAA
        * MIN for AF, AAF
    - Multirow scenario layout (paper-ready)
    """

    json_folder = Path(json_folder)
    json_files = list(json_folder.glob("**/*.json"))

    if len(json_files) == 0:
        raise ValueError("No JSON files found in the specified folder.")

    data = defaultdict(dict)

    # --- Load and filter data ---
    for file in json_files:
        with open(file, "r") as f:
            content = json.load(f)

        file_dataset = content.get("dataset", None)
        if dataset_filter is not None:
            if str(file_dataset) != str(dataset_filter):
                continue

        if "metrics" not in content or "scenario_pattern" not in content:
            continue

        strategy = content.get("strategy_name", "").lower()
        if strategy == "":
            continue

        scenario = "+".join(map(str, content["scenario_pattern"]))
        metrics = content["metrics"]

        data[scenario][strategy] = {
            "AA":  metrics["average_accuracy"],
            "AAA": metrics["average_attack_accuracy"],
            "AF":  metrics["forgetting_measure"],
            "AAF": metrics["forgetting_attack_measure"],
        }

    if len(data) == 0:
        raise ValueError("No matching JSON files after dataset filtering.")

    # Sort scenarios logically
    sorted_scenarios = sorted(
        data.keys(),
        key=lambda s: (sum(map(int, s.split("+"))), s)
    )

    strategies = ["icarl", "der", "er"]
    metrics_order = ["AA", "AAA", "AF", "AAF"]

    # Metrics where higher is better vs lower is better
    maximize_metrics = {"AA", "AAA"}
    minimize_metrics = {"AF", "AAF"}

    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")

    latex.append("\\begin{tabular}{llccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Scenario} & \\textbf{Metric} & \\textbf{iCaRL} & \\textbf{DER} & \\textbf{ER} \\\\")
    latex.append("\\midrule")

    for s_idx, scenario in enumerate(sorted_scenarios):
        scenario_data = data[scenario]

        for m_idx, metric in enumerate(metrics_order):
            row = []

            # Multirow for scenario label
            if m_idx == 0:
                row.append(f"\\multirow{{4}}{{*}}{{{scenario}}}")
            else:
                row.append("")

            row.append(metric)

            # Collect available values for best selection
            values = {}
            for strat in strategies:
                if strat in scenario_data:
                    values[strat] = scenario_data[strat][metric]

            # Determine best value
            best_value = None
            if values:
                if metric in maximize_metrics:
                    best_value = max(values.values())
                elif metric in minimize_metrics:
                    best_value = min(values.values())

            # Fill row values with bold formatting
            for strat in strategies:
                if strat in scenario_data:
                    val = scenario_data[strat][metric]
                    formatted = f"{val:.4f}"

                    # Bold best (with tolerance for float precision)
                    if best_value is not None and math.isclose(val, best_value, rel_tol=1e-6):
                        formatted = f"\\textbf{{{formatted}}}"

                    row.append(formatted)
                else:
                    row.append("--")

            latex.append(" & ".join(row) + " \\\\")

        if s_idx != len(sorted_scenarios) - 1:
            latex.append("\\midrule")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    if dataset_filter is not None:
        latex.append(
            f"\\caption{{Continual Learning Performance across Scenarios (Dataset: {dataset_filter})}}"
        )
    else:
        latex.append("\\caption{Continual Learning Performance across Scenarios}")
    latex.append("\\label{tab:cil_results}")
    latex.append("\\end{table}")

    return "\n".join(latex)