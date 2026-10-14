"""Evaluate saved predictions and reproduce the existing tables, tests and plots.

The supplied evaluator is imported unchanged. Invalid SQL scores zero here;
its loader and exact row-multiset comparison are used for executable queries.
No model inference is performed by this script.
"""

import csv
import importlib.util
import itertools
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TASK4 = Path(__file__).resolve().parent
ASSIGNMENTS = TASK4.parent
TASK2 = ASSIGNMENTS / "Task_2"
TASK3 = ASSIGNMENTS / "Task_3"
RESULTS = TASK2 / "results"
FRAMEWORKS = ("race", "clear", "crisp")
COLORS = ("#2874A6", "#D68910", "#239B56")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


provided = load_module("provided_evaluator", TASK2 / "data/evaluate_retrieval_predictions.py")
metric = load_module("new_metric", TASK3 / "metric.py")
paired = load_module("paired_statistics", TASK4 / "paired_statistics.py")


def evaluate(records):
    csv.field_size_limit(1_000_000_000)
    with (TASK2 / "data/restaurants_dataset.csv").open(encoding="utf-8", newline="") as source:
        labels = list(csv.DictReader(source))
    if len(labels) != 155 or len(records) != 465:
        raise ValueError("The full dataset and all 465 predictions are required")
    keyed = {(r["framework"], r["question_id"]): r for r in records}
    if len(keyed) != 465:
        raise ValueError("Duplicate prediction IDs")
    connection = provided.load_database()
    connection.execute("PRAGMA query_only = ON")
    rows = []
    try:
        for index, label in enumerate(labels, start=1):
            expected = json.loads(label["result"])
            for framework in FRAMEWORKS:
                record = keyed[framework, index]
                if record["question"] != label["question"]:
                    raise ValueError("Prediction is misaligned with its label")
                predicted = None
                column_count = None
                error = None
                try:
                    if record["status"] != "ok":
                        raise ValueError("Infrastructure failure: " + str(record["error"]))
                    cursor = connection.execute(record["predicted_sql"])
                    column_count = len(cursor.description or ())
                    predicted = cursor.fetchall()
                except (sqlite3.Error, ValueError) as exc:
                    error = str(exc)
                correct = predicted is not None and provided.row_multiset(predicted) == provided.row_multiset(expected)
                f1 = metric.name_multiset_f1(predicted, expected, column_count=column_count)
                rows.append({
                    "question_id": index, "framework": framework,
                    "question": label["question"], "correct": int(correct),
                    "name_multiset_f1": f1, "expected_row_count": len(expected),
                    "predicted_row_count": None if predicted is None else len(predicted),
                    "selected_column_count": column_count,
                    "sql_execution_ok": int(predicted is not None),
                    "sql_only_format": int(record["sql_only_format"]),
                    "sql_error": error, "request_status": record["status"],
                    "elapsed_seconds": record["elapsed_seconds"],
                    "attempt_count": record["attempt_count"],
                    "prompt_eval_count": record.get("prompt_eval_count"),
                    "eval_count": record.get("eval_count"),
                    "ollama_total_seconds": None if record.get("total_duration") is None else record["total_duration"] / 1e9,
                    "ollama_load_seconds": None if record.get("load_duration") is None else record["load_duration"] / 1e9,
                    "ollama_prompt_seconds": None if record.get("prompt_eval_duration") is None else record["prompt_eval_duration"] / 1e9,
                    "ollama_generation_seconds": None if record.get("eval_duration") is None else record["eval_duration"] / 1e9,
                })
    finally:
        connection.close()
    return rows


def summarize(rows):
    summary = []
    for framework in FRAMEWORKS:
        selected = [r for r in rows if r["framework"] == framework]
        nonempty = [r for r in selected if r["expected_row_count"] > 0]
        empty = [r for r in selected if r["expected_row_count"] == 0]
        times = [r["elapsed_seconds"] for r in selected if r["elapsed_seconds"] is not None]
        summary.append({
            "framework": framework, "n": len(selected),
            "correct": sum(r["correct"] for r in selected),
            "execution_accuracy": float(np.mean([r["correct"] for r in selected])),
            "mean_name_multiset_f1": float(np.mean([r["name_multiset_f1"] for r in selected])),
            "nonempty_n": len(nonempty), "nonempty_correct": sum(r["correct"] for r in nonempty),
            "nonempty_accuracy": float(np.mean([r["correct"] for r in nonempty])),
            "nonempty_mean_f1": float(np.mean([r["name_multiset_f1"] for r in nonempty])),
            "empty_n": len(empty), "empty_correct": sum(r["correct"] for r in empty),
            "empty_accuracy": float(np.mean([r["correct"] for r in empty])),
            "mean_request_seconds": float(np.mean(times)),
            "median_request_seconds": float(np.median(times)),
            "mean_generation_tokens": float(np.mean([r["eval_count"] for r in selected if r["eval_count"] is not None])),
            "sql_execution_failures": sum(not r["sql_execution_ok"] for r in selected),
            "wrong_projection_outputs": sum(r["sql_execution_ok"] and r["selected_column_count"] != 2 for r in selected),
            "empty_correct_with_wrong_projection": sum(r["correct"] and r["selected_column_count"] != 2 for r in selected),
            "format_failures": sum(not r["sql_only_format"] for r in selected),
            "request_failures": sum(r["request_status"] != "ok" for r in selected),
            "retried_requests": sum(r["attempt_count"] > 1 for r in selected),
            "partial_matches": sum(0 < r["name_multiset_f1"] < 1 for r in selected),
        })
    return summary


def matrices(rows):
    keyed = {(r["framework"], r["question_id"]): r for r in rows}
    result = {}
    for name, key in (("accuracy", "correct"), ("f1", "name_multiset_f1"),
                      ("latency", "elapsed_seconds")):
        result[name] = np.array([[keyed[f, i][key] for f in FRAMEWORKS]
                                 for i in range(1, 156)], dtype=float)
    return result


def statistical_analysis(values):
    # Fixed tests/families; no test is selected to obtain a desired p-value.
    global_tests = {
        "accuracy": paired.cochran_q(values["accuracy"]),
        "f1": paired.friedman(values["f1"], seed=43),
        "latency": paired.friedman(values["latency"], seed=44),
    }
    adjusted = paired.holm([v["permutation_p"] for v in global_tests.values()])
    for result, adjusted_p in zip(global_tests.values(), adjusted):
        result["holm_p_three_global_tests"] = adjusted_p
    comparisons = []
    for name, matrix in values.items():
        for a, b in itertools.combinations(range(3), 2):
            result = paired.paired_binary(matrix[:, a], matrix[:, b]) if name == "accuracy" else paired.paired_sign(matrix[:, a], matrix[:, b])
            comparisons.append({"metric": name, "a": FRAMEWORKS[a], "b": FRAMEWORKS[b],
                                **result, "mean_a_minus_b": float(np.mean(matrix[:, a] - matrix[:, b])),
                                "median_a_minus_b": float(np.median(matrix[:, a] - matrix[:, b]))})
    adjusted = paired.holm([r["p"] for r in comparisons])
    for result, adjusted_p in zip(comparisons, adjusted):
        result["holm_p_nine_pairwise_tests"] = adjusted_p
    return {"alpha": 0.05, "global": global_tests, "pairwise": comparisons,
            "seeds": {"accuracy": 42, "f1": 43, "latency": 44},
            "global_family": "Three overall metric tests",
            "pairwise_family": "All nine pre-specified pairwise metric tests",
            "latency_sign_direction": "Positive a-minus-b means A is slower"}


def figures(rows, summary, values):
    for directory in (TASK2 / "figures", TASK3 / "figures", TASK4 / "figures"):
        directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 140,
                         "axes.spines.top": False, "axes.spines.right": False})
    names = [f.upper() for f in FRAMEWORKS]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    accuracy = [s["execution_accuracy"] * 100 for s in summary]
    axes[0].bar(names, accuracy, color=COLORS)
    axes[0].set(ylim=(0, 100), ylabel="Exact-result accuracy (%)", title="155 questions per prompt")
    for index, value in enumerate(accuracy):
        axes[0].text(index, value + 2, f'{summary[index]["correct"]}/155', ha="center")
    axes[1].boxplot(values["latency"], tick_labels=names, showfliers=True)
    axes[1].set(ylabel="Request time (seconds)", title="Measured full-run latency")
    fig.savefig(TASK2 / "figures/accuracy_latency.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    full_f1 = [s["mean_name_multiset_f1"] for s in summary]
    nonempty_f1 = [s["nonempty_mean_f1"] for s in summary]
    x = np.arange(3)
    axes[0].bar(x - .18, full_f1, width=.36, color=COLORS, label="All questions")
    axes[0].bar(x + .18, nonempty_f1, width=.36, color=COLORS, alpha=.4,
                hatch="//", label="Nonempty references")
    axes[0].set(xticks=x, xticklabels=names, ylim=(0, 1), ylabel="Mean name-multiset F1")
    for index in range(3):
        axes[0].text(index - .18, full_f1[index] + .025, f"{full_f1[index]:.3f}", ha="center", fontsize=8)
        axes[0].text(index + .18, nonempty_f1[index] + .025, f"{nonempty_f1[index]:.3f}", ha="center", fontsize=8)
    axes[0].legend(fontsize=8)
    for index, framework in enumerate(FRAMEWORKS):
        scores = np.sort(values["f1"][:, index])
        axes[1].step(scores, np.arange(1, 156) / 155, where="post",
                     label=framework.upper(), color=COLORS[index])
    axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel="Name-multiset F1", ylabel="Fraction of questions at or below score")
    axes[1].legend()
    fig.savefig(TASK3 / "figures/name_multiset_f1.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 4.5), layout="constrained")
    axes[0].imshow(values["accuracy"].T, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                   extent=(.5, 155.5, 2.5, -.5), interpolation="nearest")
    axes[0].set(yticks=range(3), yticklabels=names, title="Exact-result success (dark = correct)")
    image = axes[1].imshow(values["f1"].T, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                          extent=(.5, 155.5, 2.5, -.5), interpolation="nearest")
    axes[1].set(yticks=range(3), yticklabels=names, xlabel="Question ID", title="Restaurant-name overlap")
    fig.colorbar(image, ax=axes[1], label="F1", shrink=.8)
    fig.savefig(TASK4 / "figures/question_outcomes.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for index, framework in enumerate(FRAMEWORKS):
        selected = [r for r in rows if r["framework"] == framework]
        axes[0].scatter([r["eval_count"] for r in selected], [r["elapsed_seconds"] for r in selected],
                        color=COLORS[index], alpha=.55, s=15, label=framework.upper())
        axes[1].plot(range(1, 156), values["latency"][:, index], color=COLORS[index],
                     alpha=.7, linewidth=.8, label=framework.upper())
    axes[0].set(xlabel="Generated tokens", ylabel="Request time (seconds)", title="Output length and latency")
    axes[1].set(xlabel="Question ID / run block", ylabel="Request time (seconds)", title="Latency over the run")
    axes[0].legend(fontsize=8)
    fig.savefig(TASK4 / "figures/latency_diagnostics.png")
    plt.close(fig)


def main():
    records = []
    integer_fields = ("question_id", "attempt_count", "total_duration", "load_duration",
                      "prompt_eval_duration", "eval_duration", "prompt_eval_count", "eval_count")
    csv.field_size_limit(1_000_000_000)
    for framework in FRAMEWORKS:
        with (RESULTS / f"predictions_{framework}.csv").open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                for field in integer_fields:
                    row[field] = int(row[field]) if row[field] else None
                row["elapsed_seconds"] = float(row["elapsed_seconds"]) if row["elapsed_seconds"] else None
                row["sql_only_format"] = row["sql_only_format"] == "True"
                records.append(row)
    rows = evaluate(records)
    summary = summarize(rows)
    values = matrices(rows)
    if not all(np.isfinite(matrix).all() for matrix in values.values()):
        raise ValueError("Missing timings for paired latency analysis")
    tests = statistical_analysis(values)
    fields = ("question_id", "framework", "correct", "name_multiset_f1", "expected_row_count",
              "predicted_row_count", "selected_column_count", "sql_execution_ok", "sql_error")
    write_csv(RESULTS / "per_question.csv", [{key: row[key] for key in fields} for row in rows])
    write_csv(RESULTS / "summary.csv", summary)
    write_json(TASK4 / "results/statistics.json", tests)
    figures(rows, summary, values)
    for row in summary:
        print(f"{row['framework'].upper()}: {row['correct']}/155, "
              f"F1={row['mean_name_multiset_f1']:.4f}, "
              f"mean request={row['mean_request_seconds']:.3f}s")


if __name__ == "__main__":
    main()

