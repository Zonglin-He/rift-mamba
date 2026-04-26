"""Relational coefficient extraction: alpha_b(q)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable
import hashlib
import math

import numpy as np

from rift_mamba.basis import RelationalBasis
from rift_mamba.records import ReachableRecord, RecordStore, TaskRow
from rift_mamba.time import days_between, parse_time


@dataclass(frozen=True)
class CoefficientMatrix:
    """Dense coefficient and missingness arrays for a task table."""

    values: np.ndarray
    masks: np.ndarray
    bases: tuple[RelationalBasis, ...]
    task_rows: tuple[TaskRow, ...]

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    @property
    def basis_names(self) -> tuple[str, ...]:
        return tuple(basis.name for basis in self.bases)


class CoefficientExtractor:
    """Compute leakage-safe relational basis coefficients."""

    def __init__(self, store: RecordStore, bases: Iterable[RelationalBasis]) -> None:
        self.store = store
        self.bases = tuple(bases)

    def transform(self, task_rows: Iterable[TaskRow], target_table: str | None = None) -> CoefficientMatrix:
        tasks = tuple(task_rows)
        values = np.zeros((len(tasks), len(self.bases)), dtype=np.float32)
        masks = np.zeros((len(tasks), len(self.bases)), dtype=bool)

        for task_index, task in enumerate(tasks):
            cache: dict[tuple[str, object], tuple[ReachableRecord, ...]] = {}
            for basis in self.bases:
                key = (basis.route.name, basis.window)
                rows = cache.get(key)
                if rows is None:
                    rows = self.store.reachable(
                        basis.route,
                        task,
                        target_table=target_table,
                        window=basis.window,
                    )
                    cache[key] = rows
                value, present = aggregate_basis(rows, basis, task.cutoff)
                values[task_index, basis.index] = value
                masks[task_index, basis.index] = present

        return CoefficientMatrix(values=values, masks=masks, bases=self.bases, task_rows=tasks)


def aggregate_basis(
    rows: tuple[ReachableRecord, ...],
    basis: RelationalBasis,
    cutoff: datetime,
) -> tuple[float, bool]:
    """Aggregate one basis for one task sample."""

    if basis.aggregator == "count":
        return float(len(rows)), True
    if not rows or basis.column_name is None:
        return 0.0, False

    raw_values = [reachable.row.get(basis.column_name) for reachable in rows]
    if basis.column_kind == "numeric":
        return _aggregate_numeric(raw_values, rows, basis.aggregator)
    if basis.column_kind == "categorical":
        return _aggregate_categorical(raw_values, rows, basis.aggregator)
    if basis.column_kind == "boolean":
        pairs = [
            (float(value), row)
            for value, row in zip(raw_values, rows, strict=False)
            if isinstance(value, bool)
        ]
        return _aggregate_number_pairs(pairs, basis.aggregator)
    if basis.column_kind == "text":
        return _aggregate_text(raw_values, rows, basis.aggregator)
    if basis.column_kind == "datetime":
        return _aggregate_datetime(raw_values, rows, basis.aggregator, cutoff)
    return 0.0, False


def _aggregate_numeric(values: list[Any], rows: tuple[ReachableRecord, ...], agg: str) -> tuple[float, bool]:
    pairs: list[tuple[float, ReachableRecord]] = []
    for value, row in zip(values, rows, strict=False):
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(number):
            pairs.append((number, row))
    return _aggregate_number_pairs(pairs, agg)


def _aggregate_number_list(
    numbers: list[float],
    rows: tuple[ReachableRecord, ...],
    agg: str,
) -> tuple[float, bool]:
    pairs = list(zip(numbers, rows, strict=False))
    return _aggregate_number_pairs(pairs, agg)


def _aggregate_number_pairs(
    pairs: list[tuple[float, ReachableRecord]],
    agg: str,
) -> tuple[float, bool]:
    numbers = [value for value, _ in pairs]
    if not numbers:
        return 0.0, False
    array = np.asarray(numbers, dtype=np.float32)
    if agg == "mean":
        return float(array.mean()), True
    if agg == "sum":
        return float(array.sum()), True
    if agg == "min":
        return float(array.min()), True
    if agg == "max":
        return float(array.max()), True
    if agg == "std":
        return float(array.std(ddof=0)), True
    if agg == "last":
        return float(_last_by_event_time(pairs)), True
    return 0.0, False


def _aggregate_categorical(values: list[Any], rows: tuple[ReachableRecord, ...], agg: str) -> tuple[float, bool]:
    pairs = [(str(value), row) for value, row in zip(values, rows, strict=False) if value is not None]
    present = [value for value, _ in pairs]
    if not present:
        return 0.0, False
    if agg == "nunique":
        return float(len(set(present))), True
    if agg == "mode_hash":
        counts: dict[str, int] = {}
        for value in present:
            counts[value] = counts.get(value, 0) + 1
        mode = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        return stable_hash_unit(mode), True
    if agg == "last_hash":
        return stable_hash_unit(str(_last_by_event_time(pairs))), True
    return 0.0, False


def _aggregate_text(values: list[Any], rows: tuple[ReachableRecord, ...], agg: str) -> tuple[float, bool]:
    pairs = [
        (str(value), row)
        for value, row in zip(values, rows, strict=False)
        if value is not None and str(value)
    ]
    texts = [value for value, _ in pairs]
    if not texts:
        return 0.0, False
    if agg == "mean_length":
        return float(np.mean([len(text) for text in texts])), True
    if agg == "last_length":
        return float(len(str(_last_by_event_time(pairs)))), True
    if agg == "last_hash":
        return stable_hash_unit(str(_last_by_event_time(pairs))), True
    return 0.0, False


def _aggregate_datetime(
    values: list[Any],
    rows: tuple[ReachableRecord, ...],
    agg: str,
    cutoff: datetime,
) -> tuple[float, bool]:
    times = [parse_time(value) for value in values if value is not None]
    times = [time for time in times if time is not None and time <= cutoff]
    if not times:
        return 0.0, False
    if agg == "last_recency_days":
        return float(days_between(cutoff, max(times))), True
    return 0.0, False


def _last_by_event_time(pairs: list[tuple[Any, ReachableRecord]]) -> Any:
    pairs.sort(key=lambda item: item[1].event_time or datetime.min)
    return pairs[-1][0]


def stable_hash_unit(value: str) -> float:
    """Map a string to a deterministic scalar in [-1, 1]."""

    digest = hashlib.md5(value.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return 2.0 * (integer / float(2**64 - 1)) - 1.0
