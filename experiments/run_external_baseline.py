"""Dispatch official external baselines with a shared JSON config."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rift_mamba import ExternalBaseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, choices=["graphsage_rdl", "relgnn", "relgt", "rt", "griffin"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--command", nargs="+", required=True)
    parser.add_argument("--workdir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = ExternalBaseline(name=args.name, command=tuple(args.command), workdir=args.workdir)
    baseline.run(
        {
            "dataset": args.dataset,
            "task": args.task,
            "baseline": args.name,
            "split_policy": "official_relbench",
            "metric_policy": "official_relbench_or_rift_metrics",
        },
        args.output,
    )


if __name__ == "__main__":
    main()
