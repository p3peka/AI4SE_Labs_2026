# AI4SE Labs - Week 3 submission

Reports: [Task 1](Week_03_Prompting/Assignments/Task_1/report.md),
[Task 2](Week_03_Prompting/Assignments/Task_2/report.md),
[Task 3](Week_03_Prompting/Assignments/Task_3/report.md),
[Task 4](Week_03_Prompting/Assignments/Task_4/report.md).
Final predictions and shared metadata are in `Task_2/results/` under the assignments.
Supplied assignments, evaluator, schema and reference data are unchanged.

From the repository root, with Python 3.12 or later:

```sh
python -m pip install -r Week_03_Prompting/requirements.txt
python Week_03_Prompting/Assignments/Task_4/analyze.py
```

This recalculates scores, statistics and the report figures from saved predictions.
To generate a new run, start local Ollama with `qwen2.5:3b-instruct` available:

```sh
python Week_03_Prompting/Assignments/Task_2/run_inference.py --output-dir Week_03_Prompting/Assignments/Task_2/results_new
```

Existing predictions are never overwritten. To analyse a new run, replace the
three submitted prediction CSVs and their metadata in `Task_2/results/` with
that run. Inference timings vary between runs.
