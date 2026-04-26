"""Schema objects for relational route modeling.

The objects in this module intentionally treat primary and foreign keys as
connectivity metadata. They are never exposed as continuous feature values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal


ColumnKind = Literal["numeric", "categorical", "text", "boolean", "datetime", "primary_key", "foreign_key"]


@dataclass(frozen=True)
class ColumnSchema:
    """A typed column in a relational table."""

    name: str
    kind: ColumnKind
    nullable: bool = True
    semantic_name: str | None = None

    @property
    def is_feature(self) -> bool:
        return self.kind not in {"primary_key", "foreign_key"}


@dataclass(frozen=True)
class ForeignKey:
    """A directed PK-FK link from one table column to another table primary key."""

    from_table: str
    from_column: str
    to_table: str
    to_column: str
    role: str | None = None

    @property
    def edge_name(self) -> str:
        role = self.role or f"{self.from_table}.{self.from_column}->{self.to_table}.{self.to_column}"
        return role.replace(" ", "_")


@dataclass(frozen=True)
class TableSchema:
    """Schema for one table."""

    name: str
    columns: tuple[ColumnSchema, ...]
    primary_key: str
    timestamp: str | None = None

    def __post_init__(self) -> None:
        names = {col.name for col in self.columns}
        if self.primary_key not in names:
            raise ValueError(f"primary key {self.primary_key!r} is not a column of {self.name!r}")
        if self.column(self.primary_key).kind != "primary_key":
            raise ValueError(f"primary key {self.name}.{self.primary_key} must have kind='primary_key'")
        if self.timestamp is not None and self.timestamp not in names:
            raise ValueError(f"timestamp {self.timestamp!r} is not a column of {self.name!r}")

    def column(self, name: str) -> ColumnSchema:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(f"unknown column {self.name}.{name}")

    @property
    def feature_columns(self) -> tuple[ColumnSchema, ...]:
        return tuple(col for col in self.columns if col.is_feature)


@dataclass(frozen=True)
class DatabaseSchema:
    """Relational schema with table metadata and PK-FK links."""

    tables: dict[str, TableSchema]
    foreign_keys: tuple[ForeignKey, ...] = field(default_factory=tuple)

    @classmethod
    def from_tables(
        cls,
        tables: Iterable[TableSchema],
        foreign_keys: Iterable[ForeignKey] = (),
    ) -> "DatabaseSchema":
        table_map = {table.name: table for table in tables}
        schema = cls(table_map, tuple(foreign_keys))
        schema.validate()
        return schema

    def table(self, name: str) -> TableSchema:
        try:
            return self.tables[name]
        except KeyError as exc:
            raise KeyError(f"unknown table {name!r}") from exc

    def validate(self) -> None:
        for fk in self.foreign_keys:
            if fk.from_table not in self.tables:
                raise ValueError(f"foreign key source table {fk.from_table!r} is missing")
            if fk.to_table not in self.tables:
                raise ValueError(f"foreign key target table {fk.to_table!r} is missing")
            source_column = self.table(fk.from_table).column(fk.from_column)
            target_table = self.table(fk.to_table)
            target_column = target_table.column(fk.to_column)
            if source_column.kind != "foreign_key":
                raise ValueError(f"foreign key source {fk.from_table}.{fk.from_column} must have kind='foreign_key'")
            if fk.to_column != target_table.primary_key or target_column.kind != "primary_key":
                raise ValueError(
                    f"foreign key target {fk.to_table}.{fk.to_column} must be the target table primary key"
                )

    def incident_foreign_keys(self, table_name: str) -> tuple[ForeignKey, ...]:
        return tuple(
            fk
            for fk in self.foreign_keys
            if fk.from_table == table_name or fk.to_table == table_name
        )
