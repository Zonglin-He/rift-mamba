"""Adapters for relational benchmark datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rift_mamba.records import RecordStore, TaskRow
from rift_mamba.schema import ColumnSchema, DatabaseSchema, ForeignKey, TableSchema
from rift_mamba.task import TaskSpec


@dataclass(frozen=True)
class RelationalDatasetBundle:
    schema: DatabaseSchema
    tables: Mapping[str, Sequence[Mapping[str, Any]]]
    task_rows: tuple[TaskRow, ...]
    task: TaskSpec

    def record_store(self) -> RecordStore:
        return RecordStore(self.schema, self.tables)


class RelBenchAdapter:
    """Thin boundary for RelBench/RelBench v2 loaders.

    The adapter keeps the RIFT-Mamba core independent of RelBench releases. A
    caller can either pass already materialized tables/schema, or subclass this
    adapter around the official RelBench APIs.
    """

    def __init__(self, bundle: RelationalDatasetBundle | None = None) -> None:
        self.bundle = bundle

    @classmethod
    def from_materialized(
        cls,
        schema: DatabaseSchema,
        tables: Mapping[str, Sequence[Mapping[str, Any]]],
        task_rows: Sequence[TaskRow],
        task: TaskSpec,
    ) -> "RelBenchAdapter":
        return cls(RelationalDatasetBundle(schema=schema, tables=tables, task_rows=tuple(task_rows), task=task))

    @classmethod
    def from_relbench(cls, *args, **kwargs) -> "RelBenchAdapter":
        """Best-effort loader for common RelBench-like materialized objects.

        Official RelBench APIs have changed across releases. This method accepts
        an object or dict exposing enough materialized fields to avoid a hard
        dependency on one release:

        - ``schema``: a ``DatabaseSchema`` or a mapping with ``tables`` and
          optional ``foreign_keys``.
        - ``tables``: mapping from table name to row mappings.
        - ``task_rows``: iterable of ``TaskRow`` or row mappings.
        - ``task``: a ``TaskSpec`` or mapping.
        """

        source = args[0] if args else kwargs
        schema = _get_field(source, "schema")
        tables = _get_field(source, "tables")
        task_rows = _get_field(source, "task_rows")
        task = _get_field(source, "task")
        if schema is not None and tables is not None and task_rows is not None and task is not None:
            return cls.from_materialized(
                schema=_coerce_schema(schema),
                tables=tables,
                task_rows=tuple(_coerce_task_row(row) for row in task_rows),
                task=_coerce_task_spec(task),
            )
        try:
            import relbench  # type: ignore  # noqa: F401
        except Exception as exc:
            raise ImportError(
                "RelBench is not installed, and the provided object did not expose "
                "schema, tables, task_rows and task. Install the official relbench "
                "package or pass a materialized object/dict with those fields."
            ) from exc
        raise ValueError("RelBench object must expose schema, tables, task_rows and task")

    def load(self) -> RelationalDatasetBundle:
        if self.bundle is None:
            raise RuntimeError("adapter has no materialized bundle")
        return self.bundle


def _get_field(source: object, name: str):
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def _coerce_schema(schema) -> DatabaseSchema:
    if isinstance(schema, DatabaseSchema):
        return schema
    if not isinstance(schema, Mapping):
        raise TypeError("schema must be DatabaseSchema or mapping")
    tables = []
    for table_data in schema.get("tables", ()):
        if isinstance(table_data, TableSchema):
            tables.append(table_data)
            continue
        columns = tuple(
            column if isinstance(column, ColumnSchema) else ColumnSchema(**column)
            for column in table_data["columns"]
        )
        tables.append(
            TableSchema(
                name=table_data["name"],
                columns=columns,
                primary_key=table_data["primary_key"],
                timestamp=table_data.get("timestamp"),
            )
        )
    foreign_keys = tuple(
        fk if isinstance(fk, ForeignKey) else ForeignKey(**fk)
        for fk in schema.get("foreign_keys", ())
    )
    return DatabaseSchema.from_tables(tables, foreign_keys)


def _coerce_task_row(row) -> TaskRow:
    if isinstance(row, TaskRow):
        return row
    if isinstance(row, Mapping):
        return TaskRow(**row)
    raise TypeError("task_rows must contain TaskRow or mappings")


def _coerce_task_spec(task) -> TaskSpec:
    if isinstance(task, TaskSpec):
        return task
    if isinstance(task, Mapping):
        return TaskSpec(**task)
    raise TypeError("task must be TaskSpec or mapping")
