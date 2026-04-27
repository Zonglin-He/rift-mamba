"""Summarize experiment result JSON files into a CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--output", default="experiments/results/summary.csv")
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.results_dir).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        if metrics:
            for metric, value in metrics.items():
                rows.append(
                    {
                        "file": path.name,
                        "model": payload.get("model"),
                        "dataset": payload.get("dataset"),
                        "task": payload.get("task"),
                        "metric": metric,
                        "value": value,
                    }
                )
        else:
            rows.append(
                {
                    "file": path.name,
                    "model": payload.get("model"),
                    "dataset": payload.get("dataset"),
                    "task": payload.get("task"),
                    "metric": "",
                    "value": "",
                }
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "model", "dataset", "task", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
