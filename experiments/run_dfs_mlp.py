"""Train a DFS-coefficient MLP baseline on RelBench splits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from experiments.common import (
    encode_labels,
    evaluate_predictions,
    load_relbench_bundle,
    output_dim_and_labels,
    predict_numpy,
    supervised_loss,
    write_json,
)
from rift_mamba import DFSMLPBaseline, ExperimentConfig, RiftDataset, prepare_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--relbench-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--backend", default="duckdb")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--include-test-labels", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--output", default="experiments/results/dfs_mlp.json")
    args = parser.parse_args()

    bundle = load_relbench_bundle(args)
    train_rows = tuple(bundle.split_rows.get("train", bundle.task_rows))
    val_rows = tuple(bundle.split_rows.get("val", ()))
    train, val = prepare_experiment(bundle, train_rows, val_rows, ExperimentConfig(max_hops=3))
    output_dim, labels, label_mapping = output_dim_and_labels(bundle.task.task_type, [row.label for row in train_rows])
    dataset = RiftDataset(train.coefficients, labels=labels.numpy(), label_dtype=labels.dtype)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = DFSMLPBaseline(train.coefficients.values.shape[1], output_dim, hidden_dim=args.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    loss_fn = supervised_loss(bundle.task.task_type)

    for _ in range(args.epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["alpha"], batch["alpha_mask"])
            loss = loss_fn(logits, batch["label"])
            loss.backward()
            optimizer.step()

    metrics = {}
    if val is not None and val_rows:
        model.eval()
        with torch.no_grad():
            logits = model(
                torch.from_numpy(val.coefficients.values).float(),
                torch.from_numpy(val.coefficients.masks).bool(),
            )
        pred = predict_numpy(bundle.task.task_type, logits)
        val_labels = [row.label for row in val_rows]
        if label_mapping is not None:
            val_labels = encode_labels(val_labels, label_mapping).numpy()
        metrics = evaluate_predictions(bundle.task.task_type, val_labels, pred)
    write_json(
        args.output,
        {
            "model": "dfs_mlp",
            "dataset": args.dataset,
            "task": args.task,
            "backend": args.backend,
            "metrics": metrics,
            "num_bases": len(train.coefficients.bases),
        },
    )


if __name__ == "__main__":
    main()
