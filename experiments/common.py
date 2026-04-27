"""Shared experiment helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rift_mamba.metrics import (
    accuracy_score,
    average_precision_score,
    mae,
    rmse,
    roc_auc_score,
)
from rift_mamba.relbench_v1_loader import load_relbench_v1
from rift_mamba.relbench_v2_loader import load_relbench_v2


def load_relbench_bundle(args):
    loader = load_relbench_v2 if args.relbench_version == "v2" else load_relbench_v1
    return loader(
        args.dataset,
        args.task,
        split="train",
        download=not args.no_download,
        backend=args.backend,
        include_test_labels=args.include_test_labels,
    )


def output_dim_and_labels(task_type: str, labels) -> tuple[int, torch.Tensor, dict[Any, int] | None]:
    array = np.asarray(labels)
    if task_type == "regression":
        return 1, torch.as_tensor(array, dtype=torch.float32).view(-1, 1), None
    if task_type == "binary_classification":
        return 2, torch.as_tensor(array, dtype=torch.long), None
    classes = sorted(set(array.tolist()))
    mapping = {value: index for index, value in enumerate(classes)}
    return len(classes), encode_labels(labels, mapping), mapping


def encode_labels(labels, mapping: dict[Any, int] | None) -> torch.Tensor:
    if mapping is None:
        return torch.as_tensor(np.asarray(labels), dtype=torch.long)
    encoded = np.asarray([mapping[value] for value in labels], dtype=np.int64)
    return torch.as_tensor(encoded, dtype=torch.long)


def supervised_loss(task_type: str):
    if task_type == "regression":
        return torch.nn.MSELoss()
    return torch.nn.CrossEntropyLoss()


def predict_numpy(task_type: str, logits: torch.Tensor) -> np.ndarray:
    if task_type == "regression":
        return logits.detach().cpu().numpy().reshape(-1)
    if task_type == "binary_classification":
        return torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
    return torch.argmax(logits, dim=-1).detach().cpu().numpy()


def evaluate_predictions(task_type: str, labels, pred) -> dict[str, float]:
    if labels is None or len(labels) == 0:
        return {}
    if task_type == "regression":
        return {"mae": mae(labels, pred), "rmse": rmse(labels, pred)}
    if task_type == "binary_classification":
        hard = np.asarray(pred) >= 0.5
        return {
            "accuracy": accuracy_score(labels, hard.astype(int)),
            "auroc": roc_auc_score(labels, pred),
            "average_precision": average_precision_score(labels, pred),
        }
    return {"accuracy": accuracy_score(labels, pred)}


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
