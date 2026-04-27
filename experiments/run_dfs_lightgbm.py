"""Train a LightGBM baseline on RIFT relational coefficients."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import evaluate_predictions, load_relbench_bundle, write_json
from rift_mamba import ExperimentConfig, prepare_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--relbench-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--backend", default="duckdb")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--include-test-labels", action="store_true")
    parser.add_argument("--output", default="experiments/results/dfs_lightgbm.json")
    args = parser.parse_args()

    try:
        import lightgbm as lgb
    except Exception as exc:
        raise SystemExit("Install lightgbm to run this baseline: pip install lightgbm") from exc

    bundle = load_relbench_bundle(args)
    train_rows = tuple(bundle.split_rows.get("train", bundle.task_rows))
    val_rows = tuple(bundle.split_rows.get("val", ()))
    train, val = prepare_experiment(bundle, train_rows, val_rows, ExperimentConfig(max_hops=3))
    y_train = [row.label for row in train_rows]
    task_type = bundle.task.task_type
    if task_type == "regression":
        model = lgb.LGBMRegressor(n_estimators=500)
    elif task_type == "binary_classification":
        model = lgb.LGBMClassifier(objective="binary", n_estimators=500)
    else:
        model = lgb.LGBMClassifier(objective="multiclass", n_estimators=500)
    model.fit(train.coefficients.values, y_train)

    metrics = {}
    if val is not None and val_rows:
        if task_type == "regression":
            pred = model.predict(val.coefficients.values)
        elif task_type == "binary_classification":
            pred = model.predict_proba(val.coefficients.values)[:, 1]
        else:
            pred = model.predict(val.coefficients.values)
        metrics = evaluate_predictions(task_type, [row.label for row in val_rows], pred)
    write_json(
        args.output,
        {
            "model": "dfs_lightgbm",
            "dataset": args.dataset,
            "task": args.task,
            "backend": args.backend,
            "metrics": metrics,
            "num_bases": len(train.coefficients.bases),
        },
    )


if __name__ == "__main__":
    main()
