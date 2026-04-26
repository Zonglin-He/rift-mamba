"""Task-aware pretraining utilities inspired by Task Vector Estimation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from rift_mamba.basis import RelationalBasis
from rift_mamba.coefficients import CoefficientMatrix, aggregate_basis
from rift_mamba.records import RecordStore, TaskRow


@dataclass(frozen=True)
class TaskVectorTargets:
    values: np.ndarray
    masks: np.ndarray
    bases: tuple[RelationalBasis, ...]
    task_rows: tuple[TaskRow, ...]
    horizon: timedelta


class TaskVectorTargetBuilder:
    """Build future-window set statistics for TVE-style pretraining."""

    def __init__(self, store: RecordStore, bases: Iterable[RelationalBasis], horizon: timedelta) -> None:
        if horizon.total_seconds() <= 0:
            raise ValueError("horizon must be positive")
        self.store = store
        self.bases = tuple(bases)
        self.horizon = horizon

    def transform(self, task_rows: Iterable[TaskRow], target_table: str | None = None) -> TaskVectorTargets:
        tasks = tuple(task_rows)
        values = np.zeros((len(tasks), len(self.bases)), dtype=np.float32)
        masks = np.zeros((len(tasks), len(self.bases)), dtype=bool)
        for task_index, task in enumerate(tasks):
            start = task.cutoff
            end = start + self.horizon
            cache = {}
            for basis in self.bases:
                rows = cache.get(basis.route.name)
                if rows is None:
                    rows = self.store.reachable_between(basis.route, task, start, end, target_table=target_table)
                    cache[basis.route.name] = rows
                value, present = aggregate_basis(rows, basis, end)
                values[task_index, basis.index] = value
                masks[task_index, basis.index] = present
        return TaskVectorTargets(values=values, masks=masks, bases=self.bases, task_rows=tasks, horizon=self.horizon)


class TaskVectorHead(nn.Module):
    """Projection head for predicting future task vectors from model features."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


def task_vector_cosine_loss(predicted: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """Cosine TVE loss over present future statistics."""

    if mask is not None:
        mask_f = mask.to(dtype=predicted.dtype)
        predicted = predicted * mask_f
        target = target * mask_f
    cosine = F.cosine_similarity(predicted, target, dim=-1, eps=1e-8)
    return (1.0 - cosine).mean()
