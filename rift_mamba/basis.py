"""Relational basis definitions.

Each basis is indexed by route, column, aggregation, and optional time window:

    b = (r, c, a, w)

The learned model later composes scalar coefficients from these bases into a
dense neural signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable

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
