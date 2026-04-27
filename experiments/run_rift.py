"""Train RIFT-Mamba on an official RelBench task."""

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
from rift_mamba import BASIS_MODES, ExperimentConfig, RiftDataset, RiftMambaModel, prepare_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--relbench-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--backend", default="duckdb")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--include-test-labels", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--sequence-max-len", type=int, default=128)
    parser.add_argument("--basis-mode", choices=sorted(BASIS_MODES), default="route_set")
    parser.add_argument("--output", default="experiments/results/rift.json")
    args = parser.parse_args()

    bundle = load_relbench_bundle(args)
    if bundle.task.task_type == "link_prediction":
        raise SystemExit("Use a pair-training script for link_prediction tasks.")
    train_rows = tuple(bundle.split_rows.get("train", bundle.task_rows))
    val_rows = tuple(bundle.split_rows.get("val", ()))
    train, val = prepare_experiment(
        bundle,
        train_rows=train_rows,
        eval_rows=val_rows,
        config=ExperimentConfig(max_hops=args.max_hops, sequence_max_len=args.sequence_max_len),
    )
    output_dim, labels, label_mapping = output_dim_and_labels(bundle.task.task_type, [row.label for row in train_rows])
    dataset = RiftDataset(train.coefficients, labels=labels.numpy(), sequences=train.sequences, label_dtype=labels.dtype)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = RiftMambaModel(
        bases=train.coefficients.bases,
        d_model=args.d_model,
        output_dim=output_dim,
        event_dim=train.sequences.values.shape[-1],
        num_temporal_routes=train.sequences.values.shape[1],
        basis_mode=args.basis_mode,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    loss_fn = supervised_loss(bundle.task.task_type)

    for _ in range(args.epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["alpha"], batch["alpha_mask"], batch.get("events"), batch.get("event_mask"))
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
                torch.from_numpy(val.sequences.values).float(),
                torch.from_numpy(val.sequences.masks).bool(),
            )
        pred = predict_numpy(bundle.task.task_type, logits)
        val_labels = [row.label for row in val_rows]
        if label_mapping is not None:
            val_labels = encode_labels(val_labels, label_mapping).numpy()
        metrics = evaluate_predictions(bundle.task.task_type, val_labels, pred)

    write_json(
        args.output,
        {
            "model": "rift_mamba",
            "dataset": args.dataset,
            "task": args.task,
            "backend": args.backend,
            "metrics": metrics,
            "num_bases": len(train.coefficients.bases),
            "num_routes": train.sequences.values.shape[1],
        },
    )


if __name__ == "__main__":
    main()
