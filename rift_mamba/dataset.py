"""Torch dataset wrappers for RIFT-Mamba tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from rift_mamba.coefficients import CoefficientMatrix
from rift_mamba.pretraining import TaskVectorTargets
from rift_mamba.sequences import SequenceBatch


class RiftDataset(Dataset):
    def __init__(
        self,
        coefficients: CoefficientMatrix,
        labels: Sequence[Any] | np.ndarray | None = None,
        sequences: SequenceBatch | None = None,
        label_dtype: torch.dtype = torch.long,
        task_vector_targets: TaskVectorTargets | None = None,
        task_vector_sample_weights: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        self.coefficients = coefficients
        self.sequences = sequences
        self.task_vector_targets = task_vector_targets
        self.task_vector_values = None
        self.task_vector_masks = None
        if labels is None:
            labels = [task.label for task in coefficients.task_rows]
        self.labels = torch.as_tensor(np.asarray(labels), dtype=label_dtype)
        if len(self.labels) != coefficients.values.shape[0]:
            raise ValueError("labels length must match coefficient rows")
        if sequences is not None and sequences.values.shape[0] != coefficients.values.shape[0]:
            raise ValueError("sequence rows must match coefficient rows")
        if sequences is not None and sequences.task_rows != coefficients.task_rows:
            raise ValueError("sequence task rows must align exactly with coefficient task rows")
        self.task_vector_sample_weights = None
        if task_vector_targets is not None:
            if task_vector_targets.values.shape[0] != coefficients.values.shape[0]:
                raise ValueError("task vector rows must match coefficient rows")
            if task_vector_targets.task_rows != coefficients.task_rows:
                raise ValueError("task vector task rows must align exactly with coefficient task rows")
            self.task_vector_values = torch.from_numpy(task_vector_targets.training_matrix()).float()
            self.task_vector_masks = torch.from_numpy(task_vector_targets.training_mask()).bool()
            if task_vector_sample_weights is not None:
                weights = torch.as_tensor(np.asarray(task_vector_sample_weights), dtype=torch.float32)
                if weights.shape[0] != coefficients.values.shape[0]:
                    raise ValueError("task vector sample weights length must match coefficient rows")
                self.task_vector_sample_weights = weights

    def __len__(self) -> int:
        return self.coefficients.values.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "alpha": torch.from_numpy(self.coefficients.values[index]).float(),
            "alpha_mask": torch.from_numpy(self.coefficients.masks[index]).bool(),
            "label": self.labels[index],
        }
        if self.sequences is not None:
            item["events"] = torch.from_numpy(self.sequences.values[index]).float()
            item["event_mask"] = torch.from_numpy(self.sequences.masks[index]).bool()
        if self.task_vector_targets is not None:
            item["task_vector_target"] = self.task_vector_values[index]
            item["task_vector_mask"] = self.task_vector_masks[index]
            if self.task_vector_sample_weights is not None:
                item["task_vector_weight"] = self.task_vector_sample_weights[index]
        return item
