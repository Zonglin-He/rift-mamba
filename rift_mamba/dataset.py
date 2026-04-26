"""Torch dataset wrappers for RIFT-Mamba tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from rift_mamba.coefficients import CoefficientMatrix
from rift_mamba.sequences import SequenceBatch


class RiftDataset(Dataset):
    def __init__(
        self,
        coefficients: CoefficientMatrix,
        labels: Sequence[Any] | np.ndarray | None = None,
        sequences: SequenceBatch | None = None,
        label_dtype: torch.dtype = torch.long,
    ) -> None:
        self.coefficients = coefficients
        self.sequences = sequences
        if labels is None:
            labels = [task.label for task in coefficients.task_rows]
        self.labels = torch.as_tensor(np.asarray(labels), dtype=label_dtype)
        if len(self.labels) != coefficients.values.shape[0]:
            raise ValueError("labels length must match coefficient rows")
        if sequences is not None and sequences.values.shape[0] != coefficients.values.shape[0]:
            raise ValueError("sequence rows must match coefficient rows")
        if sequences is not None and sequences.task_rows != coefficients.task_rows:
            raise ValueError("sequence task rows must align exactly with coefficient task rows")

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
        return item
