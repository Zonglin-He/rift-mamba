"""Relational basis definitions.

Each basis is indexed by route, column, aggregation, and optional time window:

    b = (r, c, a, w)

The learned model later composes scalar coefficients from these bases into a
dense neural signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Iterable

from rift_mamba.routes import SchemaRoute
from rift_mamba.schema import ColumnKind, DatabaseSchema


@dataclass(frozen=True)
class BasisConfig:
    """Controls which relational basis coefficients are generated."""

    windows: tuple[timedelta | None, ...] = (None, timedelta(days=7), timedelta(days=30), timedelta(days=90))
    numeric_aggs: tuple[str, ...] = ("mean", "sum", "min", "max", "std", "last")
    categorical_aggs: tuple[str, ...] = ("nunique", "mode_hash", "last_hash")
    boolean_aggs: tuple[str, ...] = ("mean", "sum", "last")
    text_aggs: tuple[str, ...] = ("mean_length", "last_length", "last_hash")
    datetime_aggs: tuple[str, ...] = ("last_recency_days",)
    include_route_count: bool = True


@dataclass(frozen=True)
class RelationalBasis:
    """A scalar relational coefficient specification."""

    index: int
    route: SchemaRoute
    aggregator: str
    window: timedelta | None
    column_name: str | None = None
    column_kind: ColumnKind | str = "row"

    @property
    def route_name(self) -> str:
        return self.route.name

    @property
    def end_table(self) -> str:
        return self.route.end_table

    @property
    def name(self) -> str:
        window = "all" if self.window is None else f"{int(self.window.total_seconds() // 86_400)}d"
        column = self.column_name or "__row__"
        return f"{self.route.name}|{column}|{self.aggregator}|{window}"


@dataclass(frozen=True)
class CompositeRelationalBasis:
    """A path-conditional coefficient over two columns on the same route path."""

    index: int
    route: SchemaRoute
    value_table: str
    value_column: str
    value_kind: ColumnKind | str
    condition_table: str
    condition_column: str
    condition_kind: ColumnKind | str
    aggregator: str
    window: timedelta | None
    condition_value: Any | None = None
    value_occurrence: int = 0
    condition_occurrence: int = 0

    @property
    def route_name(self) -> str:
        return self.route.name

    @property
    def end_table(self) -> str:
        return self.route.end_table

    @property
    def column_name(self) -> str:
        condition = "*" if self.condition_value is None else str(self.condition_value)
        return f"{self.value_table}.{self.value_column}|{self.condition_table}.{self.condition_column}={condition}"

    @property
    def column_kind(self) -> ColumnKind | str:
        return self.value_kind

    @property
    def name(self) -> str:
        window = "all" if self.window is None else f"{int(self.window.total_seconds() // 86_400)}d"
        condition = "*" if self.condition_value is None else str(self.condition_value)
        return (
            f"{self.route.name}|{self.value_table}.{self.value_column}|"
            f"{self.condition_table}.{self.condition_column}={condition}|{self.aggregator}|{window}"
        )


@dataclass(frozen=True)
class CompositeBasisSpec:
    route: SchemaRoute
    value_table: str
    value_column: str
    condition_table: str
    condition_column: str
    aggregator: str
    window: timedelta | None
    condition_value: Any | None = None
    value_occurrence: int = 0
    condition_occurrence: int = 0


def build_basis(
    schema: DatabaseSchema,
    routes: Iterable[SchemaRoute],
    config: BasisConfig | None = None,
    exclude_columns: Iterable[tuple[str, str]] = (),
) -> tuple[RelationalBasis, ...]:
    """Generate route-column-aggregation-window bases.

    Primary and foreign keys are excluded through ``TableSchema.feature_columns``.
    The caller can also exclude task targets or proxy leakage columns.
    """

    cfg = config or BasisConfig()
    excluded = set(exclude_columns)
    bases: list[RelationalBasis] = []
    for route in routes:
        table = schema.table(route.end_table)
        windows = _windows_for_route(route, cfg.windows)
        if cfg.include_route_count:
            for window in windows:
                bases.append(
                    RelationalBasis(
                        index=len(bases),
                        route=route,
                        column_name=None,
                        column_kind="row",
                        aggregator="count",
                        window=window,
                    )
                )

        for column in table.feature_columns:
            if (table.name, column.name) in excluded:
                continue
            for agg in _aggregators_for_kind(column.kind, cfg):
                for window in windows:
                    bases.append(
                        RelationalBasis(
                            index=len(bases),
                            route=route,
                            column_name=column.name,
                            column_kind=column.kind,
                            aggregator=agg,
                            window=window,
                        )
                    )
    return tuple(bases)


def build_composite_basis(
    schema: DatabaseSchema,
    specs: Iterable[CompositeBasisSpec],
    start_index: int = 0,
    exclude_columns: Iterable[tuple[str, str]] = (),
) -> tuple[CompositeRelationalBasis, ...]:
    """Build manual path-conditional bases from composite specs."""

    excluded = set(exclude_columns)
    bases: list[CompositeRelationalBasis] = []
    for spec in specs:
        value_column = schema.table(spec.value_table).column(spec.value_column)
        condition_column = schema.table(spec.condition_table).column(spec.condition_column)
        if (spec.value_table, spec.value_column) in excluded:
            continue
        if (spec.condition_table, spec.condition_column) in excluded:
            continue
        if not value_column.is_feature or not condition_column.is_feature:
            continue
        bases.append(
            CompositeRelationalBasis(
                index=start_index + len(bases),
                route=spec.route,
                value_table=spec.value_table,
                value_column=spec.value_column,
                value_kind=value_column.kind,
                condition_table=spec.condition_table,
                condition_column=spec.condition_column,
                condition_kind=condition_column.kind,
                aggregator=spec.aggregator,
                window=spec.window,
                condition_value=spec.condition_value,
                value_occurrence=spec.value_occurrence,
                condition_occurrence=spec.condition_occurrence,
            )
        )
    return tuple(bases)


def _windows_for_route(route: SchemaRoute, windows: tuple[timedelta | None, ...]) -> tuple[timedelta | None, ...]:
    if route.hop_count == 0:
        return (None,)
    return windows


def _aggregators_for_kind(kind: ColumnKind, config: BasisConfig) -> tuple[str, ...]:
    if kind == "numeric":
        return config.numeric_aggs
    if kind == "categorical":
        return config.categorical_aggs
    if kind == "boolean":
        return config.boolean_aggs
    if kind == "text":
        return config.text_aggs
    if kind == "datetime":
        return config.datetime_aggs
    return ()
