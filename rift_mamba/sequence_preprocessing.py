"""Train-only preprocessing for route-wise event tensors."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np

from rift_mamba.sequences import SequenceBatch


class EventFeatureStandardizer:
    """Standardize selected event-token dimensions using train sequences only.

    ``TemporalSequenceBuilder`` marks scalar event dimensions that carry raw
    numeric magnitudes, such as numeric cell values and recency in days. This
    class learns mean/std on valid training events and reuses those statistics
    for validation/test batches, avoiding normalization leakage from future
    splits.
    """

    def __init__(self, feature_indices: Iterable[int] | None = None) -> None:
        self.feature_indices = None if feature_indices is None else tuple(feature_indices)
        self.present_indices_: tuple[int, ...] = ()
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, sequences: SequenceBatch) -> "EventFeatureStandardizer":
        indices = self._resolve_feature_indices(sequences)
        present_indices = self._resolve_present_indices(sequences, indices)
        means = np.zeros(len(indices), dtype=np.float32)
        scales = np.ones(len(indices), dtype=np.float32)

        for offset, (feature_index, present_index) in enumerate(zip(indices, present_indices, strict=True)):
            values = sequences.values[..., feature_index]
            valid = self._valid_mask(sequences, values, present_index)
            if not bool(valid.any()):
                continue
            present_values = values[valid].astype(np.float64, copy=False)
            means[offset] = float(present_values.mean())
            std = float(present_values.std())
            scales[offset] = 1.0 if std < 1.0e-6 else std

        self.feature_indices = indices
        self.present_indices_ = present_indices
        self.mean_ = means
        self.scale_ = scales
        return self

    def transform(self, sequences: SequenceBatch) -> SequenceBatch:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("EventFeatureStandardizer must be fit before transform")
        indices = self._resolve_feature_indices(sequences)
        if indices != self.feature_indices:
            raise ValueError("sequence feature layout does not match fitted standardizer")
        values = sequences.values.copy()
        for offset, (feature_index, present_index) in enumerate(
            zip(indices, self.present_indices_, strict=True)
        ):
            feature = values[..., feature_index]
            valid = self._valid_mask(sequences, feature, present_index)
            scaled = (feature - self.mean_[offset]) / self.scale_[offset]
            values[..., feature_index] = np.where(valid, scaled, 0.0).astype(np.float32)
        return replace(sequences, values=values)

    def fit_transform(self, sequences: SequenceBatch) -> SequenceBatch:
        return self.fit(sequences).transform(sequences)

    def _resolve_feature_indices(self, sequences: SequenceBatch) -> tuple[int, ...]:
        indices = sequences.standardize_feature_indices if self.feature_indices is None else self.feature_indices
        for index in indices:
            if index < 0 or index >= sequences.values.shape[-1]:
                raise ValueError("feature index is outside event dimension")
        return tuple(indices)

    def _resolve_present_indices(
        self,
        sequences: SequenceBatch,
        feature_indices: tuple[int, ...],
    ) -> tuple[int, ...]:
        present_indices = sequences.standardize_present_indices
        if not present_indices:
            present_indices = tuple(-1 for _ in feature_indices)
        if len(present_indices) != len(feature_indices):
            raise ValueError("standardize_present_indices must align with feature indices")
        for index in present_indices:
            if index != -1 and (index < 0 or index >= sequences.values.shape[-1]):
                raise ValueError("present index is outside event dimension")
        return tuple(present_indices)

    @staticmethod
    def _valid_mask(sequences: SequenceBatch, values: np.ndarray, present_index: int) -> np.ndarray:
        valid = sequences.masks & np.isfinite(values)
        if present_index != -1:
            valid = valid & (sequences.values[..., present_index] > 0.5)
        return valid
