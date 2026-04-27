"""Minimal supervised and TVE-auxiliary training loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import torch
from torch import Tensor

from rift_mamba.nn import RiftMambaModel
from rift_mamba.pretraining import TaskVectorHead, task_vector_cosine_loss


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    supervised_loss: float
    tve_loss: float
    num_batches: int


def train_epoch(
    model: RiftMambaModel,
    dataloader,
    optimizer: torch.optim.Optimizer,
    supervised_loss_fn: Callable[[Tensor, Tensor], Tensor],
    *,
    task_vector_head: TaskVectorHead | None = None,
    task_vector_weight: float = 0.0,
    device: torch.device | str | None = None,
) -> EpochMetrics:
    """Train one epoch with optional TVE auxiliary loss.

    Batches are expected to come from ``RiftDataset``. When ``task_vector_head``
    is provided, the batch must include ``task_vector_target`` and may include
    ``task_vector_mask`` and ``task_vector_weight``.
    """

    model.train()
    if task_vector_head is not None:
        task_vector_head.train()
    totals = _MetricTotals()

    for batch in dataloader:
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits, features = _forward_with_features(model, batch)
        supervised_loss = supervised_loss_fn(logits, batch["label"])
        tve_loss = _batch_tve_loss(task_vector_head, features, batch)
        loss = supervised_loss + float(task_vector_weight) * tve_loss
        loss.backward()
        optimizer.step()
        totals.add(loss, supervised_loss, tve_loss)

    return totals.to_metrics()


@torch.no_grad()
def evaluate_epoch(
    model: RiftMambaModel,
    dataloader,
    supervised_loss_fn: Callable[[Tensor, Tensor], Tensor],
    *,
    task_vector_head: TaskVectorHead | None = None,
    task_vector_weight: float = 0.0,
    device: torch.device | str | None = None,
) -> EpochMetrics:
    """Evaluate supervised plus optional TVE loss without parameter updates."""

    model.eval()
    if task_vector_head is not None:
        task_vector_head.eval()
    totals = _MetricTotals()

    for batch in dataloader:
        batch = _move_batch(batch, device)
        logits, features = _forward_with_features(model, batch)
        supervised_loss = supervised_loss_fn(logits, batch["label"])
        tve_loss = _batch_tve_loss(task_vector_head, features, batch)
        loss = supervised_loss + float(task_vector_weight) * tve_loss
        totals.add(loss, supervised_loss, tve_loss)

    return totals.to_metrics()


def _forward_with_features(model: RiftMambaModel, batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
    if "events" in batch:
        output = model(
            batch["alpha"],
            batch["alpha_mask"],
            batch["events"],
            batch["event_mask"],
            return_features=True,
        )
    else:
        output = model(batch["alpha"], batch["alpha_mask"], return_features=True)
    logits, features = output
    return logits, features


def _batch_tve_loss(
    task_vector_head: TaskVectorHead | None,
    features: Tensor,
    batch: Mapping[str, Tensor],
) -> Tensor:
    if task_vector_head is None:
        return features.new_zeros(())
    if "task_vector_target" not in batch:
        raise ValueError("task_vector_head requires task_vector_target in each batch")
    predicted = task_vector_head(features)
    sample_weight = batch.get("task_vector_weight")
    return task_vector_cosine_loss(
        predicted,
        batch["task_vector_target"],
        batch.get("task_vector_mask"),
        sample_weight=sample_weight,
    )


def _move_batch(batch: Mapping[str, Tensor], device: torch.device | str | None) -> dict[str, Tensor]:
    if device is None:
        return dict(batch)
    return {name: value.to(device) for name, value in batch.items()}


@dataclass
class _MetricTotals:
    loss: float = 0.0
    supervised_loss: float = 0.0
    tve_loss: float = 0.0
    num_batches: int = 0

    def add(self, loss: Tensor, supervised_loss: Tensor, tve_loss: Tensor) -> None:
        self.loss += float(loss.detach().cpu())
        self.supervised_loss += float(supervised_loss.detach().cpu())
        self.tve_loss += float(tve_loss.detach().cpu())
        self.num_batches += 1

    def to_metrics(self) -> EpochMetrics:
        denom = max(self.num_batches, 1)
        return EpochMetrics(
            loss=self.loss / denom,
            supervised_loss=self.supervised_loss / denom,
            tve_loss=self.tve_loss / denom,
            num_batches=self.num_batches,
        )
