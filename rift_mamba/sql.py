"""SQL/DataFrame pushdown coefficient extractors."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from rift_mamba.basis import CompositeRelationalBasis, RelationalBasis
from rift_mamba.coefficients import CoefficientExtractor, CoefficientMatrix
from rift_mamba.records import RecordStore, TaskRow
from rift_mamba.schema import DatabaseSchema


SUPPORTED_SQL_AGGS = {
    "count",
    "mean",
    "sum",
    "min",
    "max",
    "std",
    "last",
    "last_recency_days",
}


class DuckDBCoefficientExtractor:
    """DuckDB pushdown for route joins and scalar aggregations.

    Unsupported bases, such as text/hash aggregations and composite bases, are
    computed through the in-memory extractor and merged back into the result.
    """

    def __init__(
        self,
        schema: DatabaseSchema,
        tables: Mapping[str, Sequence[Mapping[str, Any]]],
        bases: Iterable[RelationalBasis],
    ) -> None:
        import duckdb  # type: ignore  # noqa: F401

        self.schema = schema
        self.tables = tables
        self.bases = tuple(bases)
        self.fallback = CoefficientExtractor(RecordStore(schema, tables), self.bases)

    def transform(self, task_rows: Iterable[TaskRow], target_table: str | None = None) -> CoefficientMatrix:
        tasks = tuple(task_rows)
        values = np.zeros((len(tasks), len(self.bases)), dtype=np.float32)
        masks = np.zeros((len(tasks), len(self.bases)), dtype=bool)
        if not tasks:
            return CoefficientMatrix(values=values, masks=masks, bases=self.bases, task_rows=tasks)

        fallback_indices = [index for index, basis in enumerate(self.bases) if not _is_sql_supported(basis)]
        if fallback_indices:
            fallback = self.fallback.transform(tasks, target_table=target_table)
            values[:, fallback_indices] = fallback.values[:, fallback_indices]
            masks[:, fallback_indices] = fallback.masks[:, fallback_indices]

        import duckdb
        import pandas as pd

        conn = duckdb.connect(database=":memory:")
        try:
            for table_name, rows in self.tables.items():
                frame = pd.DataFrame(list(rows))
                table_schema = self.schema.table(table_name)
                datetime_columns = [
                    column.name for column in table_schema.columns if column.kind == "datetime" or column.name == table_schema.timestamp
                ]
                for column in datetime_columns:
                    if column in frame:
                        frame[column] = pd.to_datetime(frame[column])
                conn.register(_rel_name(table_name), frame)
            task_df = pd.DataFrame(
                {
                    "__row_pos": np.arange(len(tasks), dtype=np.int64),
                    "__entity_id": [task.entity_id for task in tasks],
                    "__seed_time": pd.to_datetime([task.seed_time for task in tasks]),
                }
            )
            conn.register("__rift_tasks", task_df)
            for basis_index, basis in enumerate(self.bases):
                if basis_index in fallback_indices:
                    continue
                sql = _duckdb_basis_sql(self.schema, basis, target_table)
                result = conn.execute(sql).fetchdf().sort_values("__row_pos")
                series = result["value"].to_numpy()
                present = result["present"].to_numpy(dtype=bool)
                values[:, basis.index] = np.nan_to_num(series.astype(np.float32), nan=0.0)
                masks[:, basis.index] = present
        finally:
            conn.close()
        return CoefficientMatrix(values=values, masks=masks, bases=self.bases, task_rows=tasks)


class PolarsCoefficientExtractor:
    """Polars lazy pushdown for route joins and scalar aggregations."""

    def __init__(
        self,
        schema: DatabaseSchema,
        tables: Mapping[str, Sequence[Mapping[str, Any]]],
        bases: Iterable[RelationalBasis],
    ) -> None:
        import polars  # type: ignore  # noqa: F401

        self.schema = schema
        self.tables = tables
        self.bases = tuple(bases)
        self.fallback = CoefficientExtractor(RecordStore(schema, tables), self.bases)

    def transform(self, task_rows: Iterable[TaskRow], target_table: str | None = None) -> CoefficientMatrix:
        tasks = tuple(task_rows)
        values = np.zeros((len(tasks), len(self.bases)), dtype=np.float32)
        masks = np.zeros((len(tasks), len(self.bases)), dtype=bool)
        if not tasks:
            return CoefficientMatrix(values=values, masks=masks, bases=self.bases, task_rows=tasks)

        import polars as pl
        import pandas as pd

        fallback_indices = [index for index, basis in enumerate(self.bases) if not _is_sql_supported(basis)]
        if fallback_indices:
            fallback = self.fallback.transform(tasks, target_table=target_table)
            values[:, fallback_indices] = fallback.values[:, fallback_indices]
            masks[:, fallback_indices] = fallback.masks[:, fallback_indices]

        table_frames = {}
        for name, rows in self.tables.items():
            frame = pd.DataFrame(list(rows))
            table_schema = self.schema.table(name)
            datetime_columns = [
                column.name for column in table_schema.columns if column.kind == "datetime" or column.name == table_schema.timestamp
            ]
            for column in datetime_columns:
                if column in frame:
                    frame[column] = pd.to_datetime(frame[column])
            table_frames[name] = pl.from_pandas(frame).lazy()
        task_lf = pl.DataFrame(
            {
                "__row_pos": list(range(len(tasks))),
                "__entity_id": [task.entity_id for task in tasks],
                "__seed_time": [task.cutoff for task in tasks],
            }
        ).lazy()

        for basis_index, basis in enumerate(self.bases):
            if basis_index in fallback_indices:
                continue
            result = _polars_basis_frame(self.schema, table_frames, task_lf, basis, target_table).collect()
            result = result.sort("__row_pos")
            series = result["value"].to_numpy()
            present = result["present"].to_numpy().astype(bool)
            values[:, basis.index] = np.nan_to_num(series.astype(np.float32), nan=0.0)
            masks[:, basis.index] = present
        return CoefficientMatrix(values=values, masks=masks, bases=self.bases, task_rows=tasks)


def _is_sql_supported(basis: RelationalBasis) -> bool:
    if isinstance(basis, CompositeRelationalBasis):
        return False
    if basis.aggregator not in SUPPORTED_SQL_AGGS:
        return False
    if basis.aggregator == "last_recency_days":
        return basis.column_kind == "datetime"
    if basis.column_kind in {"row", "numeric", "boolean", "datetime"}:
        return True
    return False


def _duckdb_basis_sql(
    schema: DatabaseSchema,
    basis: RelationalBasis,
    target_table: str | None,
) -> str:
    route = basis.route
    if target_table is not None and route.start_table != target_table:
        raise ValueError(f"route starts at {route.start_table!r}, expected {target_table!r}")
    joins, event_expr = _duckdb_route_joins(schema, route)
    end_alias = f"a{route.hop_count}"
    end_table = schema.table(route.end_table)
    pk_expr = f"{end_alias}.{_q(end_table.primary_key)}"
    filter_expr = _duckdb_filter_expr(basis.window, event_expr, pk_expr)
    value_expr, present_expr = _duckdb_agg_expr(basis, end_alias, filter_expr, event_expr, pk_expr)
    return f"""
        SELECT
            t.__row_pos,
            {value_expr} AS value,
            {present_expr} AS present
        FROM __rift_tasks AS t
        {joins}
        GROUP BY t.__row_pos
        ORDER BY t.__row_pos
    """


def _duckdb_route_joins(schema: DatabaseSchema, route) -> tuple[str, str | None]:
    start = schema.table(route.start_table)
    causal = _duckdb_causal_clause(start, "a0")
    joins = [
        f"LEFT JOIN {_qrel(route.start_table)} AS a0 "
        f"ON a0.{_q(start.primary_key)} = t.__entity_id{causal}"
    ]
    event_terms = _event_terms(schema, route)
    for index, step in enumerate(route.steps, start=1):
        prev_alias = f"a{index - 1}"
        alias = f"a{index}"
        target = schema.table(step.target_table)
        causal = _duckdb_causal_clause(target, alias)
        joins.append(
            f"LEFT JOIN {_qrel(step.target_table)} AS {alias} "
            f"ON {alias}.{_q(step.target_column)} = {prev_alias}.{_q(step.source_column)}{causal}"
        )
    return "\n".join(joins), _greatest_expr(event_terms)


def _duckdb_causal_clause(table, alias: str) -> str:
    if table.timestamp is None:
        return ""
    return f" AND {alias}.{_q(table.timestamp)} <= t.__seed_time"


def _event_terms(schema: DatabaseSchema, route) -> list[str]:
    terms = []
    for index, table_name in enumerate(route.table_path):
        timestamp = schema.table(table_name).timestamp
        if timestamp is not None:
            terms.append(f"a{index}.{_q(timestamp)}")
    return terms


def _greatest_expr(terms: list[str]) -> str | None:
    if not terms:
        return None
    if len(terms) == 1:
        return terms[0]
    return f"GREATEST({', '.join(terms)})"


def _duckdb_filter_expr(window: timedelta | None, event_expr: str | None, pk_expr: str) -> str:
    parts = [f"{pk_expr} IS NOT NULL"]
    if window is not None:
        if event_expr is None:
            parts.append("FALSE")
        else:
            days = int(window.total_seconds() // 86_400)
            parts.append(f"{event_expr} > t.__seed_time - INTERVAL '{days} days'")
            parts.append(f"{event_expr} <= t.__seed_time")
    return " AND ".join(parts)


def _duckdb_agg_expr(
    basis: RelationalBasis,
    alias: str,
    filter_expr: str,
    event_expr: str | None,
    pk_expr: str,
) -> tuple[str, str]:
    if basis.aggregator == "count":
        value = f"COUNT({pk_expr}) FILTER (WHERE {filter_expr})"
        return value, "TRUE"
    if basis.column_name is None:
        return "NULL", "FALSE"
    col = f"{alias}.{_q(basis.column_name)}"
    value_filter = f"{filter_expr} AND {col} IS NOT NULL"
    if basis.aggregator == "mean":
        value = f"AVG(CAST({col} AS DOUBLE)) FILTER (WHERE {value_filter})"
    elif basis.aggregator == "sum":
        value = f"SUM(CAST({col} AS DOUBLE)) FILTER (WHERE {value_filter})"
    elif basis.aggregator == "min":
        value = f"MIN(CAST({col} AS DOUBLE)) FILTER (WHERE {value_filter})"
    elif basis.aggregator == "max":
        value = f"MAX(CAST({col} AS DOUBLE)) FILTER (WHERE {value_filter})"
    elif basis.aggregator == "std":
        value = f"STDDEV_POP(CAST({col} AS DOUBLE)) FILTER (WHERE {value_filter})"
    elif basis.aggregator == "last":
        if event_expr is None:
            value = f"MAX(CAST({col} AS DOUBLE)) FILTER (WHERE {value_filter})"
        else:
            value = f"ARG_MAX(CAST({col} AS DOUBLE), {event_expr}) FILTER (WHERE {value_filter})"
    elif basis.aggregator == "last_recency_days":
        value = f"DATE_DIFF('day', MAX({col}) FILTER (WHERE {value_filter}), ANY_VALUE(t.__seed_time))"
    else:
        value = "NULL"
    present = f"COUNT({col}) FILTER (WHERE {value_filter}) > 0"
    return value, present


def _polars_basis_frame(schema, table_frames, task_lf, basis, target_table):
    import polars as pl

    route = basis.route
    if target_table is not None and route.start_table != target_table:
        raise ValueError(f"route starts at {route.start_table!r}, expected {target_table!r}")
    frame = task_lf
    event_cols: list[str] = []
    causal_conditions = []
    for index, table_name in enumerate(route.table_path):
        alias = f"a{index}"
        table = schema.table(table_name)
        table_lf = _polars_alias(table_frames[table_name], alias)
        if index == 0:
            frame = frame.join(
                table_lf,
                left_on="__entity_id",
                right_on=f"{alias}__{table.primary_key}",
                how="left",
                coalesce=False,
            )
        else:
            step = route.steps[index - 1]
            frame = frame.join(
                table_lf,
                left_on=f"a{index - 1}__{step.source_column}",
                right_on=f"{alias}__{step.target_column}",
                how="left",
                coalesce=False,
            )
        if table.timestamp is not None:
            time_col = f"{alias}__{table.timestamp}"
            event_cols.append(time_col)
            causal_conditions.append(pl.col(time_col).is_null() | (pl.col(time_col) <= pl.col("__seed_time")))
    end_alias = f"a{route.hop_count}"
    end_pk = f"{end_alias}__{schema.table(route.end_table).primary_key}"
    valid = pl.col(end_pk).is_not_null()
    for condition in causal_conditions:
        valid = valid & condition
    if basis.window is not None:
        if not event_cols:
            valid = pl.lit(False)
        else:
            event_expr = pl.max_horizontal([pl.col(col) for col in event_cols])
            valid = valid & (event_expr <= pl.col("__seed_time")) & (
                event_expr > pl.col("__seed_time") - pl.duration(days=int(basis.window.total_seconds() // 86_400))
            )
    if basis.aggregator == "count":
        agg = pl.col(end_pk).filter(valid).count().alias("value")
        present = pl.lit(True).alias("present")
    else:
        col = f"{end_alias}__{basis.column_name}"
        col_expr = pl.col(col).cast(pl.Float64, strict=False).filter(valid & pl.col(col).is_not_null())
        if basis.aggregator == "mean":
            agg = col_expr.mean().alias("value")
        elif basis.aggregator == "sum":
            agg = col_expr.sum().alias("value")
        elif basis.aggregator == "min":
            agg = col_expr.min().alias("value")
        elif basis.aggregator == "max":
            agg = col_expr.max().alias("value")
        elif basis.aggregator == "std":
            agg = col_expr.std(ddof=0).alias("value")
        elif basis.aggregator == "last":
            agg = col_expr.last().alias("value")
        elif basis.aggregator == "last_recency_days":
            max_time = pl.col(col).filter(valid & pl.col(col).is_not_null()).max()
            agg = (pl.col("__seed_time").first() - max_time).dt.total_days().alias("value")
        else:
            agg = pl.lit(None).alias("value")
        present = (pl.col(col).filter(valid & pl.col(col).is_not_null()).count() > 0).alias("present")
    return frame.group_by("__row_pos").agg([agg, present])


def _polars_alias(lf, alias: str):
    return lf.rename({name: f"{alias}__{name}" for name in lf.collect_schema().names()})


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _qrel(identifier: str) -> str:
    return _q(_rel_name(identifier))


def _rel_name(table_name: str) -> str:
    return f"rift_table_{table_name}"
