"""In-memory relational records and leakage-safe route traversal."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence

from rift_mamba.routes import SchemaRoute
from rift_mamba.schema import DatabaseSchema, TableSchema
from rift_mamba.time import parse_time


Record = dict[str, Any]


@dataclass(frozen=True)
class TaskRow:
    """A prediction sample q=(entity, seed_time, label)."""

    row_id: str | int
    entity_id: Any
    seed_time: datetime | str
    label: Any = None

    @property
    def cutoff(self) -> datetime:
        parsed = parse_time(self.seed_time)
        if parsed is None:
            raise ValueError("task seed_time cannot be None")
        return parsed


@dataclass(frozen=True)
class ReachableRecord:
    """One row reached by a schema route for a single task sample."""

    table: str
    row: Record
    route: SchemaRoute
    event_time: datetime | None
    path: tuple[tuple[str, Any], ...]
    path_rows: tuple[tuple[str, Record], ...]


class RecordStore:
    """A small, dependency-free relational record store.

    Tables are provided as ``dict[str, Sequence[Mapping[str, Any]]]``. The
    store builds PK and column indexes for deterministic route traversal.
    """

    def __init__(self, schema: DatabaseSchema, tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        self.schema = schema
        self.tables: dict[str, list[Record]] = {
            table_name: [dict(row) for row in rows] for table_name, rows in tables.items()
        }
        for table_name in schema.tables:
            self.tables.setdefault(table_name, [])
        self._pk_index: dict[tuple[str, Any], Record] = {}
        self._column_index: dict[tuple[str, str, Any], list[Record]] = {}
        self._build_indexes()

    def _build_indexes(self) -> None:
        for table_name, rows in self.tables.items():
            table = self.schema.table(table_name)
            for row in rows:
                pk_value = row.get(table.primary_key)
                if pk_value is not None:
                    self._pk_index[(table_name, pk_value)] = row
                for col in table.columns:
                    value = row.get(col.name)
                    if _is_hashable(value):
                        self._column_index.setdefault((table_name, col.name, value), []).append(row)

    def get_by_pk(self, table_name: str, pk_value: Any) -> Record | None:
        return self._pk_index.get((table_name, pk_value))

    def rows_by_value(self, table_name: str, column: str, value: Any) -> tuple[Record, ...]:
        if not _is_hashable(value):
            return ()
        return tuple(self._column_index.get((table_name, column, value), ()))

    def reachable(
        self,
        route: SchemaRoute,
        task: TaskRow,
        target_table: str | None = None,
        window: timedelta | None = None,
    ) -> tuple[ReachableRecord, ...]:
        """Return rows reachable from ``task.entity_id`` through ``route``.

        Causal filtering is applied at every timestamped table encountered in
        the route, not only at the final table. This prevents a future fact row
        from carrying a timeless dimension row into the input.
        """

        if target_table is not None and route.start_table != target_table:
            raise ValueError(f"route starts at {route.start_table!r}, expected {target_table!r}")
        cutoff = task.cutoff
        start = self.get_by_pk(route.start_table, task.entity_id)
        if start is None or not self._is_causal_row(self.schema.table(route.start_table), start, cutoff):
            return ()

        start_key = (route.start_table, start.get(self.schema.table(route.start_table).primary_key))
        states: list[
            tuple[Record, datetime | None, tuple[tuple[str, Any], ...], tuple[tuple[str, Record], ...]]
        ] = [
            (
                start,
                self._row_time(self.schema.table(route.start_table), start),
                (start_key,),
                ((route.start_table, start),),
            )
        ]

        for step in route.steps:
            target_schema = self.schema.table(step.target_table)
            new_states: list[
                tuple[Record, datetime | None, tuple[tuple[str, Any], ...], tuple[tuple[str, Record], ...]]
            ] = []
            for row, event_time, path, path_rows in states:
                join_value = row.get(step.source_column)
                if join_value is None:
                    continue
                candidates = self.rows_by_value(step.target_table, step.target_column, join_value)
                for candidate in candidates:
                    if not self._is_causal_row(target_schema, candidate, cutoff):
                        continue
                    candidate_time = self._row_time(target_schema, candidate)
                    next_event_time = _latest_time(event_time, candidate_time)
                    key = (step.target_table, candidate.get(target_schema.primary_key))
                    new_states.append(
                        (
                            candidate,
                            next_event_time,
                            path + (key,),
                            path_rows + ((step.target_table, candidate),),
                        )
                    )
            states = new_states
            if not states:
                return ()

        if window is not None:
            states = [
                state
                for state in states
                if state[1] is not None and cutoff - window < state[1] <= cutoff
            ]

        return tuple(
            ReachableRecord(
                table=route.end_table,
                row=row,
                route=route,
                event_time=event_time,
                path=path,
                path_rows=path_rows,
            )
            for row, event_time, path, path_rows in states
        )

    def reachable_between(
        self,
        route: SchemaRoute,
        task: TaskRow,
        start_time: datetime | str,
        end_time: datetime | str,
        target_table: str | None = None,
    ) -> tuple[ReachableRecord, ...]:
        """Return rows with propagated event time in ``(start_time, end_time]``.

        This is intended for future-window targets such as task-vector
        pretraining. It should not be used to build supervised model inputs.
        """

        start = parse_time(start_time)
        end = parse_time(end_time)
        if start is None or end is None:
            raise ValueError("start_time and end_time are required")
        if end <= start:
            raise ValueError("end_time must be later than start_time")
        end_task = TaskRow(row_id=task.row_id, entity_id=task.entity_id, seed_time=end, label=task.label)
        rows = self.reachable(route, end_task, target_table=target_table)
        return tuple(
            row
            for row in rows
            if row.event_time is not None and start < row.event_time <= end
        )

    def _row_time(self, table: TableSchema, row: Mapping[str, Any]) -> datetime | None:
        if table.timestamp is None:
            return None
        return parse_time(row.get(table.timestamp))

    def _is_causal_row(self, table: TableSchema, row: Mapping[str, Any], cutoff: datetime) -> bool:
        row_time = self._row_time(table, row)
        return row_time is None or row_time <= cutoff


def _latest_time(left: datetime | None, right: datetime | None) -> datetime | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _is_hashable(value: Any) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True
