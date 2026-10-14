"""Generate the three final prediction CSVs with independent Ollama requests.

Use --output-dir with a new directory; existing results are never overwritten.
Only the question column is sent to the model.
"""

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

TASK = Path(__file__).resolve().parent
FRAMEWORKS = ("race", "clear", "crisp")
MODEL = "qwen2.5:3b-instruct"
HOST = "http://127.0.0.1:11434"
OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 4096,
           "num_predict": 512, "repeat_penalty": 1.0}
ORDER_SEED = 20261014
FIELDS = ("question_id", "question", "framework", "prompt_id", "model",
          "predicted_sql", "raw_model_output", "elapsed_seconds", "started_utc",
          "status", "error", "attempt_count", "done_reason", "sql_only_format",
          "total_duration", "load_duration", "prompt_eval_duration", "eval_duration",
          "prompt_eval_count", "eval_count")


def now():
    return datetime.now(timezone.utc).isoformat()


def api(path, payload=None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(HOST + path, data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def generate(framework, index, question, template):
    row = {"question_id": index, "question": question, "framework": framework,
           "prompt_id": framework + "_v2_house_number", "model": MODEL}
    payload = {"model": MODEL, "messages": [{"role": "user", "content":
               template.replace("{{question}}", question)}], "stream": False,
               "options": OPTIONS, "keep_alive": "30m"}
    for attempt in range(1, 4):
        started = now()
        timer = time.perf_counter()
        try:
            response = api("/api/chat", payload)
            elapsed = time.perf_counter() - timer
            if not response.get("done"):
                raise ValueError("Incomplete Ollama response")
            raw = response["message"]["content"]
            sql = raw.strip()
            row.update(predicted_sql=sql, raw_model_output=raw if raw != sql else "",
                       elapsed_seconds=elapsed, started_utc=started, status="ok",
                       error="", attempt_count=attempt, done_reason=response.get("done_reason"),
                       sql_only_format=bool(re.match(r"\ASELECT\b", sql, re.I)) and
                       not any(marker in sql for marker in ("```", "--", "/*")))
            for field in FIELDS[14:]:
                row[field] = response.get(field)
            return row
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            row.update(predicted_sql="", raw_model_output="", elapsed_seconds=None,
                       started_utc=started, status="error", error=str(error),
                       attempt_count=attempt, sql_only_format=False)
            if attempt < 3:
                time.sleep(attempt)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=TASK / "results")
    args = parser.parse_args()
    if any((args.output_dir / f"predictions_{f}.csv").exists() for f in FRAMEWORKS):
        parser.error("Prediction files already exist; use --output-dir with a new directory")
    csv.field_size_limit(1_000_000_000)
    with (TASK / "data/restaurants_dataset.csv").open(encoding="utf-8", newline="") as source:
        questions = [row["question"] for row in csv.DictReader(source)]
    assert len(questions) == 155
    templates = {f: (TASK.parent / f"Task_1/prompts/{f}.txt").read_text(encoding="utf-8")
                 for f in FRAMEWORKS}
    assert all(text.count("{{question}}") == 1 for text in templates.values())
    assert "List Italian restaurants in Berkeley with a rating above 3.25." not in questions
    model = next(m for m in api("/api/tags")["models"] if m["name"] == MODEL)
    metadata = {"provider": "Ollama", "api": HOST + "/api/chat", "model": MODEL,
                "model_digest": model["digest"], "generation_settings": OPTIONS,
                "stream": False, "keep_alive": "30m", "framework_order_seed": ORDER_SEED,
                "ollama_version": api("/api/version")["version"], "python": sys.version,
                "question_count": 155, "prediction_count": 465, "started_utc": now(),
                "prompt_sha256": {f: hashlib.sha256(t.encode()).hexdigest() for f, t in templates.items()},
                "request_context": "One user message; no previous messages or answers.",
                "timing_definition": "perf_counter wall time around each complete HTTP request",
                "ollama_duration_unit": "nanoseconds",
                "raw_output_convention": "A blank raw_model_output means it equals predicted_sql."}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for framework in FRAMEWORKS:
        with (args.output_dir / f"predictions_{framework}.csv").open("w", encoding="utf-8", newline="") as target:
            csv.DictWriter(target, fieldnames=FIELDS).writeheader()
    randomizer = random.Random(ORDER_SEED)
    completed = 0
    for index, question in enumerate(questions, 1):
        order = list(FRAMEWORKS)
        randomizer.shuffle(order)
        for framework in order:
            row = generate(framework, index, question, templates[framework])
            with (args.output_dir / f"predictions_{framework}.csv").open("a", encoding="utf-8", newline="") as target:
                csv.DictWriter(target, fieldnames=FIELDS).writerow(row)
            completed += 1
            print(f"{completed}/465 {framework} Q{index} {row['status']}", flush=True)
    metadata["completed_utc"] = now()
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
