"""Train two-tower RIFT-Mamba for RelBench recommendation/link tasks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.common import load_relbench_bundle, write_json
from rift_mamba import (
    BASIS_MODES,
    LinkExperimentConfig,
    LinkRiftDataset,
    PairRiftMambaModel,
    RiftMambaModel,
    average_precision_score,
    prepare_link_experiment,
    roc_auc_score,
    sample_negative_links,
    transform_link_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--relbench-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--backend", default="duckdb")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--include-test-labels", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--negative-ratio", type=int, default=1)
    parser.add_argument("--basis-mode", choices=sorted(BASIS_MODES), default="route_set")
    parser.add_argument("--output", default="experiments/results/rift_link.json")
    args = parser.parse_args()

    bundle = load_relbench_bundle(args)
    if bundle.task.task_type != "link_prediction":
        raise SystemExit("This script expects a link_prediction / recommendation task.")
    dst_table = bundle.schema.table(bundle.task.dst_entity_table)
    candidate_dst = [row[dst_table.primary_key] for row in bundle.tables[bundle.task.dst_entity_table]]
    train_rows = sample_negative_links(
        tuple(bundle.split_rows.get("train", bundle.task_rows)),
        candidate_dst,
        num_negatives=args.negative_ratio,
    )
    val_positive = tuple(bundle.split_rows.get("val", ()))
    val_rows = sample_negative_links(val_positive, candidate_dst, num_negatives=args.negative_ratio) if val_positive else ()
    train = prepare_link_experiment(bundle, train_rows, LinkExperimentConfig(max_hops=3))
    dataset = LinkRiftDataset(train)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    src_encoder = RiftMambaModel(
        train.src_coefficients.bases,
        d_model=args.d_model,
        output_dim=2,
        event_dim=train.src_sequences.values.shape[-1],
        num_temporal_routes=train.src_sequences.values.shape[1],
        basis_mode=args.basis_mode,
    )
    dst_encoder = RiftMambaModel(
        train.dst_coefficients.bases,
        d_model=args.d_model,
        output_dim=2,
        event_dim=train.dst_sequences.values.shape[-1],
        num_temporal_routes=train.dst_sequences.values.shape[1],
        basis_mode=args.basis_mode,
    )
    model = PairRiftMambaModel(src_encoder, dst_encoder)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(args.epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch["src_alpha"],
                batch["src_alpha_mask"],
                batch["dst_alpha"],
                batch["dst_alpha_mask"],
                batch["src_events"],
                batch["src_event_mask"],
                batch["dst_events"],
                batch["dst_event_mask"],
            )
            loss = loss_fn(logits, batch["label"].float())
            loss.backward()
            optimizer.step()

    metrics = {}
    if val_rows:
        val = transform_link_experiment(bundle, val_rows, train)
        val_dataset = LinkRiftDataset(val)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        y_true, y_score = [], []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    batch["src_alpha"],
                    batch["src_alpha_mask"],
                    batch["dst_alpha"],
                    batch["dst_alpha_mask"],
                    batch["src_events"],
                    batch["src_event_mask"],
                    batch["dst_events"],
                    batch["dst_event_mask"],
                )
                y_score.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                y_true.extend(batch["label"].cpu().numpy().tolist())
        metrics = {
            "auroc": roc_auc_score(np.asarray(y_true), np.asarray(y_score)),
            "average_precision": average_precision_score(np.asarray(y_true), np.asarray(y_score)),
        }
    write_json(
        args.output,
        {
            "model": "rift_mamba_link",
            "dataset": args.dataset,
            "task": args.task,
            "backend": args.backend,
            "metrics": metrics,
            "negative_ratio": args.negative_ratio,
        },
    )


if __name__ == "__main__":
    main()
