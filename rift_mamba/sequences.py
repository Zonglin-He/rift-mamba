"""Route-wise causal event sequence construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from rift_mamba.coefficients import stable_hash_unit
from rift_mamba.records import ReachableRecord, RecordStore, TaskRow
from rift_mamba.routes import SchemaRoute
from rift_mamba.schema import ColumnSchema, DatabaseSchema
from rift_mamba.time import days_between, parse_time


@dataclass(frozen=True)
class EventColumn:
    table: str
    column: str
    kind: str

    @property
    def name(self) -> str:
        return f"{self.table}.{self.column}"


@dataclass(frozen=True)
class SequenceBatch:
    """Padded event tensors for route-wise sequence encoders."""

    values: np.ndarray
    masks: np.ndarray
    routes: tuple[SchemaRoute, ...]
    event_columns: tuple[EventColumn, ...]
    task_rows: tuple[TaskRow, ...]

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.values.shape


class TemporalSequenceBuilder:
    """Build route-wise event sequences S_r(q) with tau <= t0."""

    def __init__(
        self,
        schema: DatabaseSchema,
        store: RecordStore,
        routes: Iterable[SchemaRoute],
        max_len: int = 128,
        event_columns: Iterable[EventColumn] | None = None,
    ) -> None:
        if max_len <= 0:
            raise ValueError("max_len must be positive")
        self.schema = schema
        self.store = store
        self.routes = tuple(routes)
        self.max_len = max_len
        self.event_columns = tuple(event_columns) if event_columns is not None else self._infer_event_columns()

    @property
    def event_dim(self) -> int:
        return len(self.event_columns) * 2 + 2

    def transform(self, task_rows: Iterable[TaskRow], target_table: str | None = None) -> SequenceBatch:
        tasks = tuple(task_rows)
        values = np.zeros((len(tasks), len(self.routes), self.max_len, self.event_dim), dtype=np.float32)
        masks = np.zeros((len(tasks), len(self.routes), self.max_len), dtype=bool)

        for task_index, task in enumerate(tasks):
            cutoff = task.cutoff
            for route_index, route in enumerate(self.routes):
                rows = [
                    row
                    for row in self.store.reachable(route, task, target_table=target_table)
                    if row.event_time is not None and row.event_time <= cutoff
                ]
                rows.sort(key=lambda row: row.event_time)
                rows = rows[-self.max_len :]
                offset = self.max_len - len(rows)
                for item_index, reachable in enumerate(rows, start=offset):
                    values[task_index, route_index, item_index] = self._encode_event(reachable, task)
                    masks[task_index, route_index, item_index] = True

        return SequenceBatch(
            values=values,
            masks=masks,
            routes=self.routes,
            event_columns=self.event_columns,
            task_rows=tasks,
        )

    def _infer_event_columns(self) -> tuple[EventColumn, ...]:
        seen: set[tuple[str, str]] = set()
        columns: list[EventColumn] = []
        for route in self.routes:
            table = self.schema.table(route.end_table)
            for column in table.feature_columns:
                key = (table.name, column.name)
                if key in seen:
                    continue
                seen.add(key)
                columns.append(EventColumn(table=table.name, column=column.name, kind=column.kind))
        return tuple(columns)

    def _encode_event(self, reachable: ReachableRecord, task: TaskRow) -> np.ndarray:
        vector = np.zeros(self.event_dim, dtype=np.float32)
        for column_index, event_column in enumerate(self.event_columns):
            if event_column.table != reachable.table:
                continue
            value = reachable.row.get(event_column.column)
            encoded, present = encode_value(value, event_column.kind, task.cutoff)
            vector[2 * column_index] = encoded
            vector[2 * column_index + 1] = float(present)

        base = len(self.event_columns) * 2
        if reachable.event_time is not None:
            vector[base] = float(days_between(task.cutoff, reachable.event_time))
        vector[base + 1] = float(reachable.route.hop_count)
        return vector


def encode_value(value: Any, kind: str, cutoff) -> tuple[float, bool]:
    if value is None:
        return 0.0, False
    if kind == "numeric":
        try:
            return float(value), True
        except (TypeError, ValueError):
            return 0.0, False
    if kind == "boolean":
        return float(bool(value)), True
    if kind in {"categorical", "text"}:
        return stable_hash_unit(str(value)), True
    if kind == "datetime":
        parsed = parse_time(value)
        if parsed is None or parsed > cutoff:
            return 0.0, False
        return float(days_between(cutoff, parsed)), True
    return 0.0, False
