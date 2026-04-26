"""Storage backend interfaces for large relational experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, Any

from rift_mamba.coefficients import CoefficientExtractor, CoefficientMatrix
from rift_mamba.records import RecordStore, TaskRow
from rift_mamba.schema import DatabaseSchema


class CoefficientBackend(Protocol):
    name: str

    def coefficient_extractor(self, bases) -> CoefficientExtractor:
        ...


@dataclass
class InMemoryBackend:
    """Default backend backed by ``RecordStore``."""

    schema: DatabaseSchema
    tables: Mapping[str, Sequence[Mapping[str, Any]]]
    name: str = "materialized"

    def record_store(self) -> RecordStore:
        return RecordStore(self.schema, self.tables)

    def coefficient_extractor(self, bases) -> CoefficientExtractor:
        return CoefficientExtractor(self.record_store(), bases)


class DuckDBBackend(InMemoryBackend):
    """DuckDB boundary for large-table materialization and future SQL pushdown."""

    name = "duckdb"

    def __init__(self, schema: DatabaseSchema, tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        try:
            import duckdb  # type: ignore  # noqa: F401
        except Exception as exc:
            raise ImportError("DuckDBBackend requires optional dependency duckdb; install rift-mamba[backends].") from exc
        super().__init__(schema=schema, tables=tables, name="duckdb")


class PolarsBackend(InMemoryBackend):
    """Polars boundary for columnar preprocessing before coefficient extraction."""

    name = "polars"

    def __init__(self, schema: DatabaseSchema, tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        try:
            import polars  # type: ignore  # noqa: F401
        except Exception as exc:
            raise ImportError("PolarsBackend requires optional dependency polars; install rift-mamba[backends].") from exc
        super().__init__(schema=schema, tables=tables, name="polars")


def make_backend(
    name: str,
    schema: DatabaseSchema,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> InMemoryBackend:
    normalized = name.lower()
    if normalized in {"materialized", "memory", "inmemory"}:
        return InMemoryBackend(schema, tables)
    if normalized == "duckdb":
        return DuckDBBackend(schema, tables)
    if normalized == "polars":
        return PolarsBackend(schema, tables)
    if normalized in {"duckdb-polars", "polars-duckdb"}:
        return DuckDBBackend(schema, tables)
    raise ValueError(f"unknown backend {name!r}")
