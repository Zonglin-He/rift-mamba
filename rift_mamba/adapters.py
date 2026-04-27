"""Adapters for relational benchmark datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from rift_mamba.records import LinkTaskRow, RecordStore, TaskRow
from rift_mamba.backends import InMemoryBackend, make_backend
from rift_mamba.schema import ColumnSchema, DatabaseSchema, ForeignKey, TableSchema
from rift_mamba.task import TaskSpec


TaskLike = TaskRow | LinkTaskRow


@dataclass(frozen=True)
class RelationalDatasetBundle:
    schema: DatabaseSchema
    tables: Mapping[str, Sequence[Mapping[str, Any]]]
    task_rows: tuple[TaskLike, ...]
    task: TaskSpec
    backend: str = "materialized"
    split_rows: Mapping[str, tuple[TaskLike, ...]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def record_store(self) -> RecordStore:
        return RecordStore(self.schema, self.tables)

    def coefficient_backend(self) -> InMemoryBackend:
        return make_backend(self.backend, self.schema, self.tables)


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
        task_rows: Sequence[TaskLike],
        task: TaskSpec,
        backend: str = "materialized",
        split_rows: Mapping[str, Sequence[TaskLike]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "RelBenchAdapter":
        return cls(
            RelationalDatasetBundle(
                schema=schema,
                tables=tables,
                task_rows=tuple(task_rows),
                task=task,
                backend=backend,
                split_rows={key: tuple(value) for key, value in (split_rows or {}).items()},
                metadata=dict(metadata or {}),
            )
        )

    @classmethod
    def from_official(
        cls,
        dataset_name: str,
        task_name: str,
        *,
        split: str | None = None,
        download: bool = True,
        backend: str = "materialized",
        include_test_labels: bool = False,
        upto_test_timestamp: bool = True,
    ) -> "RelBenchAdapter":
        """Load an official RelBench dataset/task through the public API."""

        try:
            from relbench.datasets import get_dataset
            from relbench.tasks import get_task
        except Exception as exc:
            raise ImportError(
                "Official RelBench loading requires the relbench package. "
                "Install it with `pip install relbench` or `pip install -e .[relbench]`."
            ) from exc
        dataset = get_dataset(dataset_name, download=download)
        task = get_task(dataset_name, task_name, download=download)
        return cls.from_relbench_objects(
            dataset,
            task,
            dataset_name=dataset_name,
            task_name=task_name,
            split=split,
            backend=backend,
            include_test_labels=include_test_labels,
            upto_test_timestamp=upto_test_timestamp,
        )

    @classmethod
    def from_relbench_objects(
        cls,
        dataset,
        task,
        *,
        dataset_name: str | None = None,
        task_name: str | None = None,
        split: str | None = None,
        backend: str = "materialized",
        include_test_labels: bool = False,
        upto_test_timestamp: bool = True,
    ) -> "RelBenchAdapter":
        """Adapt already instantiated official RelBench dataset/task objects."""

        db = dataset.get_db(upto_test_timestamp=upto_test_timestamp)
        schema, tables = _coerce_relbench_database(db)
        split_rows = {
            name: _task_rows_from_relbench_task(
                task,
                name,
                include_labels=(name != "test" or include_test_labels),
            )
            for name in ("train", "val", "test")
        }
        selected_split = split or "train"
        if selected_split not in split_rows:
            raise ValueError("split must be one of 'train', 'val' or 'test'")
        task_spec = _task_spec_from_relbench_task(task)
        return cls.from_materialized(
            schema=schema,
            tables=tables,
            task_rows=split_rows[selected_split],
            task=task_spec,
            backend=backend,
            split_rows=split_rows,
            metadata={
                "relbench_dataset_name": dataset_name,
                "relbench_task_name": task_name,
                "relbench_dataset": dataset,
                "relbench_task": task,
                "val_timestamp": getattr(dataset, "val_timestamp", None),
                "test_timestamp": getattr(dataset, "test_timestamp", None),
            },
        )

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
        backend = _get_field(source, "backend") or "materialized"
        if schema is not None and tables is not None and task_rows is not None and task is not None:
            return cls.from_materialized(
                schema=_coerce_schema(schema),
                tables=tables,
                task_rows=tuple(_coerce_task_like(row) for row in task_rows),
                task=_coerce_task_spec(task),
                backend=backend,
                split_rows={
                    key: tuple(_coerce_task_like(row) for row in value)
                    for key, value in (_get_field(source, "split_rows") or {}).items()
                },
                metadata=_get_field(source, "metadata") or {},
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


def _coerce_task_like(row) -> TaskLike:
    if isinstance(row, TaskRow):
        return row
    if isinstance(row, LinkTaskRow):
        return row
    if isinstance(row, Mapping):
        if "src_id" in row and "dst_id" in row:
            return LinkTaskRow(**row)
        return TaskRow(**row)
    raise TypeError("task_rows must contain TaskRow, LinkTaskRow or mappings")


def _coerce_task_spec(task) -> TaskSpec:
    if isinstance(task, TaskSpec):
        return task
    if isinstance(task, Mapping):
        return TaskSpec(**task)
    raise TypeError("task must be TaskSpec or mapping")


def _coerce_relbench_database(db) -> tuple[DatabaseSchema, Mapping[str, Sequence[Mapping[str, Any]]]]:
    table_dict = getattr(db, "table_dict", None)
    if table_dict is None:
        raise TypeError("RelBench database object must expose table_dict")
    tables: list[TableSchema] = []
    rows_by_table: dict[str, list[dict[str, Any]]] = {}
    synthetic_pkeys: dict[str, str] = {}

    for table_name, rel_table in table_dict.items():
        df = rel_table.df.copy()
        pkey_col = rel_table.pkey_col
        if pkey_col is None:
            pkey_col = "__rift_row_id__"
            df.insert(0, pkey_col, range(len(df)))
            synthetic_pkeys[table_name] = pkey_col
        fkeys = set(rel_table.fkey_col_to_pkey_table.keys())
        columns = []
        for column_name in df.columns:
            if column_name == pkey_col:
                kind = "primary_key"
            elif column_name in fkeys:
                kind = "foreign_key"
            elif column_name == rel_table.time_col:
                kind = "datetime"
            else:
                kind = _infer_column_kind(df[column_name], str(column_name))
            columns.append(ColumnSchema(str(column_name), kind))
        tables.append(
            TableSchema(
                name=table_name,
                columns=tuple(columns),
                primary_key=pkey_col,
                timestamp=rel_table.time_col,
            )
        )
        rows_by_table[table_name] = _dataframe_to_records(df)

    foreign_keys: list[ForeignKey] = []
    for table_name, rel_table in table_dict.items():
        for fkey_col, pkey_table in rel_table.fkey_col_to_pkey_table.items():
            target = table_dict[pkey_table]
            target_pkey = target.pkey_col or synthetic_pkeys.get(pkey_table)
            if target_pkey is None:
                raise ValueError(f"target table {pkey_table!r} has no primary key")
            foreign_keys.append(
                ForeignKey(
                    from_table=table_name,
                    from_column=fkey_col,
                    to_table=pkey_table,
                    to_column=target_pkey,
                    role=fkey_col,
                )
            )

    return DatabaseSchema.from_tables(tables, foreign_keys), rows_by_table


def _dataframe_to_records(df) -> list[dict[str, Any]]:
    cleaned = df.astype(object).where(df.notna(), None)
    return [dict(row) for row in cleaned.to_dict(orient="records")]


def _infer_column_kind(series, column_name: str) -> str:
    try:
        from pandas.api.types import (
            is_bool_dtype,
            is_datetime64_any_dtype,
            is_numeric_dtype,
            is_string_dtype,
        )
    except Exception as exc:
        raise ImportError("RelBench dataframe conversion requires pandas") from exc
    if is_bool_dtype(series):
        return "boolean"
    if is_datetime64_any_dtype(series):
        return "datetime"
    if is_numeric_dtype(series):
        return "numeric"
    lowered = column_name.lower()
    if any(token in lowered for token in ("text", "description", "comment", "summary", "title", "body")):
        return "text"
    non_null = series.dropna()
    if is_string_dtype(series) and len(non_null) > 0:
        avg_len = non_null.astype(str).str.len().mean()
        if avg_len >= 64:
            return "text"
    return "categorical"


def _task_spec_from_relbench_task(task) -> TaskSpec:
    task_type = getattr(getattr(task, "task_type", None), "value", getattr(task, "task_type", "unknown"))
    if hasattr(task, "src_entity_table") and hasattr(task, "dst_entity_table"):
        return TaskSpec(
            target_table=task.src_entity_table,
            entity_column=task.src_entity_col,
            seed_time_column=task.time_col,
            label_column=task.dst_entity_col,
            task_type=task_type,
            prediction_horizon=_coerce_timedelta(getattr(task, "timedelta", None)),
            src_entity_table=task.src_entity_table,
            src_entity_column=task.src_entity_col,
            dst_entity_table=task.dst_entity_table,
            dst_entity_column=task.dst_entity_col,
            eval_k=getattr(task, "eval_k", None),
        )
    leakage_columns = tuple(tuple(item) for item in getattr(task, "remove_columns", ()) or ())
    return TaskSpec(
        target_table=task.entity_table,
        entity_column=task.entity_col,
        seed_time_column=task.time_col,
        label_column=task.target_col,
        task_type=task_type,
        prediction_horizon=_coerce_timedelta(getattr(task, "timedelta", None)),
        target_column=(task.entity_table, task.target_col),
        leakage_columns=leakage_columns,
    )


def _task_rows_from_relbench_task(task, split: str, include_labels: bool) -> tuple[TaskLike, ...]:
    table = task.get_table(split, mask_input_cols=not include_labels)
    df = table.df
    if hasattr(task, "src_entity_col") and hasattr(task, "dst_entity_col"):
        rows: list[LinkTaskRow] = []
        for row_pos, row in df.reset_index(drop=True).iterrows():
            dst_values = row.get(task.dst_entity_col, None)
            if isinstance(dst_values, (list, tuple, set)):
                iterable = tuple(dst_values)
            elif dst_values is None:
                iterable = ()
            else:
                iterable = (dst_values,)
            for dst_pos, dst_id in enumerate(iterable):
                rows.append(
                    LinkTaskRow(
                        row_id=f"{split}:{row_pos}:{dst_pos}",
                        src_id=row[task.src_entity_col],
                        dst_id=dst_id,
                        seed_time=row[task.time_col],
                        label=1,
                    )
                )
        return tuple(rows)

    rows = []
    for row_pos, row in df.reset_index(drop=True).iterrows():
        rows.append(
            TaskRow(
                row_id=f"{split}:{row_pos}",
                entity_id=row[task.entity_col],
                seed_time=row[task.time_col],
                label=row[task.target_col] if include_labels and task.target_col in row else None,
            )
        )
    return tuple(rows)


def _coerce_timedelta(value):
    if value is None:
        return None
    if hasattr(value, "to_pytimedelta"):
        return value.to_pytimedelta()
    return value
