"""Route-wise causal event sequence construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from rift_mamba.coefficients import stable_hash_unit
from rift_mamba.records import ReachableRecord, RecordStore, TaskRow
from rift_mamba.routes import SchemaRoute
from rift_mamba.schema import DatabaseSchema
from rift_mamba.semantic import SchemaSemanticEncoder
from rift_mamba.time import days_between, fourier_time_features, parse_time


@dataclass(frozen=True)
class EventColumn:
    table: str
    column: str
    kind: str
    occurrence: int = 0

    @property
    def name(self) -> str:
        suffix = "" if self.occurrence == 0 else f"#{self.occurrence}"
        return f"{self.table}{suffix}.{self.column}"


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
        semantic_encoder: SchemaSemanticEncoder | None = None,
        value_embedding_dim: int = 8,
        time_periods: tuple[float, ...] = (1.0, 7.0, 30.0, 365.0),
        exclude_columns: Iterable[tuple[str, str]] = (),
    ) -> None:
        if max_len <= 0:
            raise ValueError("max_len must be positive")
        if value_embedding_dim < 0:
            raise ValueError("value_embedding_dim must be non-negative")
        self.schema = schema
        self.store = store
        self.routes = tuple(routes)
        self.max_len = max_len
        self.semantic_encoder = semantic_encoder or SchemaSemanticEncoder()
        self.value_embedding_dim = value_embedding_dim
        self.time_periods = time_periods
        self.exclude_columns = set(exclude_columns)
        if event_columns is None:
            self.event_columns = self._infer_event_columns()
        else:
            self.event_columns = tuple(
                column
                for column in event_columns
                if (column.table, column.column) not in self.exclude_columns
            )

    @property
    def event_dim(self) -> int:
        per_column = 2 + self.value_embedding_dim
        return len(self.event_columns) * per_column + 1 + 2 * len(self.time_periods) + 1 + self.semantic_encoder.dim

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
        seen: set[tuple[str, str, int]] = set()
        columns: list[EventColumn] = []
        for route in self.routes:
            occurrences: dict[str, int] = {}
            for table_name in route.table_path:
                occurrence = occurrences.get(table_name, 0)
                occurrences[table_name] = occurrence + 1
                table = self.schema.table(table_name)
                for column in table.feature_columns:
                    if (table.name, column.name) in self.exclude_columns:
                        continue
                    key = (table.name, column.name, occurrence)
                    if key in seen:
                        continue
                    seen.add(key)
                    columns.append(
                        EventColumn(
                            table=table.name,
                            column=column.name,
                            kind=column.kind,
                            occurrence=occurrence,
                        )
                    )
        return tuple(columns)

    def _encode_event(self, reachable: ReachableRecord, task: TaskRow) -> np.ndarray:
        vector = np.zeros(self.event_dim, dtype=np.float32)
        per_column = 2 + self.value_embedding_dim
        path_rows: dict[tuple[str, int], dict] = {}
        occurrences: dict[str, int] = {}
        for table_name, row in reachable.path_rows:
            occurrence = occurrences.get(table_name, 0)
            occurrences[table_name] = occurrence + 1
            path_rows[(table_name, occurrence)] = row
        for column_index, event_column in enumerate(self.event_columns):
            row = path_rows.get((event_column.table, event_column.occurrence))
            if row is None:
                continue
            value = row.get(event_column.column)
            scalar, embedding, present = encode_value(
                value,
                event_column.kind,
                task.cutoff,
                event_column.table,
                event_column.column,
                self.semantic_encoder,
                self.value_embedding_dim,
            )
            base = per_column * column_index
            vector[base] = scalar
            if self.value_embedding_dim:
                vector[base + 1 : base + 1 + self.value_embedding_dim] = embedding
            vector[base + per_column - 1] = float(present)

        base = len(self.event_columns) * per_column
        delta_days = 0.0
        if reachable.event_time is not None:
            delta_days = float(days_between(task.cutoff, reachable.event_time))
        vector[base] = delta_days
        vector[base + 1 : base + 1 + 2 * len(self.time_periods)] = fourier_time_features(
            delta_days,
            self.time_periods,
        )
        vector[base + 1 + 2 * len(self.time_periods)] = float(reachable.route.hop_count)
        semantic_base = base + 2 + 2 * len(self.time_periods)
        vector[semantic_base : semantic_base + self.semantic_encoder.dim] = self.semantic_encoder.encode_route(
            reachable.route
        )
        return vector


def encode_value(
    value: Any,
    kind: str,
    cutoff,
    table: str = "",
    column: str = "",
    semantic_encoder: SchemaSemanticEncoder | None = None,
    value_embedding_dim: int = 0,
) -> tuple[float, np.ndarray, bool]:
    embedding = np.zeros(value_embedding_dim, dtype=np.float32)
    if value is None:
        return 0.0, embedding, False
    if kind == "numeric":
        try:
            return float(value), embedding, True
        except (TypeError, ValueError):
            return 0.0, embedding, False
    if kind == "boolean":
        return float(bool(value)), embedding, True
    if kind in {"categorical", "text"}:
        encoder = semantic_encoder or SchemaSemanticEncoder()
        encoded = encoder.encode_category(table, column, value)
        if value_embedding_dim:
            embedding[: min(value_embedding_dim, encoded.shape[0])] = encoded[:value_embedding_dim]
        return 0.0, embedding, True
    if kind == "datetime":
        parsed = parse_time(value)
        if parsed is None or parsed > cutoff:
            return 0.0, embedding, False
        return float(days_between(cutoff, parsed)), embedding, True
    return stable_hash_unit(str(value)), embedding, True
