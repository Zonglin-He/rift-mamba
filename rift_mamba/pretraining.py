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
    null_indicators: np.ndarray
    bases: tuple[RelationalBasis, ...]
    task_rows: tuple[TaskRow, ...]
    horizon: timedelta

    def training_matrix(self, include_null_indicator: bool = True) -> np.ndarray:
        if include_null_indicator:
            return np.concatenate([self.values, self.null_indicators.astype(np.float32)], axis=1)
        return self.values

    def training_mask(self, include_null_indicator: bool = True) -> np.ndarray:
        if include_null_indicator:
            return np.ones((self.values.shape[0], self.values.shape[1] * 2), dtype=bool)
        return self.masks

    def sample_weights(self, mode: str = "inverse_present") -> np.ndarray:
        present = self.masks.sum(axis=1).astype(np.float32)
        if mode == "none":
            return np.ones_like(present, dtype=np.float32)
        if mode == "inverse_present":
            return 1.0 / np.maximum(present, 1.0)
        if mode == "sqrt_inverse_present":
            return 1.0 / np.sqrt(np.maximum(present, 1.0))
        raise ValueError("unknown sample weight mode")


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
        null_indicators = ~masks
        return TaskVectorTargets(
            values=values,
            masks=masks,
            null_indicators=null_indicators,
            bases=self.bases,
            task_rows=tasks,
            horizon=self.horizon,
        )


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


def task_vector_cosine_loss(
    predicted: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    sample_weight: Tensor | None = None,
) -> Tensor:
    """Cosine TVE loss over present future statistics."""

    if mask is not None:
        mask_f = mask.to(dtype=predicted.dtype)
        predicted = predicted * mask_f
        target = target * mask_f
    cosine = F.cosine_similarity(predicted, target, dim=-1, eps=1e-8)
    loss = 1.0 - cosine
    if mask is not None:
        valid = mask.any(dim=-1)
        loss = torch.where(valid, loss, torch.zeros_like(loss))
    if sample_weight is not None:
        weights = sample_weight.to(dtype=loss.dtype)
        loss = loss * weights
        denom = weights.sum().clamp_min(1e-8)
        return loss.sum() / denom
    if mask is not None:
        denom = mask.any(dim=-1).to(dtype=loss.dtype).sum().clamp_min(1.0)
        return loss.sum() / denom
    return loss.mean()
