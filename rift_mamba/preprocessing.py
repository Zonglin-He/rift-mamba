"""Preprocessing utilities for relational coefficients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rift_mamba.coefficients import CoefficientMatrix


@dataclass
class CoefficientStandardizer:
    """Fit train-only normalization statistics for alpha_b(q)."""

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, matrix: CoefficientMatrix) -> "CoefficientStandardizer":
        values = matrix.values
        masks = matrix.masks
        mean = np.zeros(values.shape[1], dtype=np.float32)
        scale = np.ones(values.shape[1], dtype=np.float32)
        for col in range(values.shape[1]):
            present = values[masks[:, col], col]
            if present.size == 0:
                continue
            mean[col] = float(present.mean())
            std = float(present.std())
            scale[col] = std if std > 1e-6 else 1.0
        self.mean_ = mean
        self.scale_ = scale
        return self

    def transform(self, matrix: CoefficientMatrix) -> CoefficientMatrix:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("CoefficientStandardizer must be fit before transform")
        values = (matrix.values - self.mean_) / self.scale_
        values = np.where(matrix.masks, values, 0.0).astype(np.float32)
        return CoefficientMatrix(values=values, masks=matrix.masks.copy(), bases=matrix.bases, task_rows=matrix.task_rows)

    def fit_transform(self, matrix: CoefficientMatrix) -> CoefficientMatrix:
        return self.fit(matrix).transform(matrix)
