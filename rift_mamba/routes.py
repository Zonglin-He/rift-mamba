"""Schema route enumeration.

Routes are the semantic unit used by RIFT-Mamba. A route step may traverse a
PK-FK link either forward or backward, but downstream modules treat the whole
route as a basis index rather than performing edge-level message passing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rift_mamba.schema import DatabaseSchema, ForeignKey


Direction = Literal["forward", "backward"]


@dataclass(frozen=True)
class RouteStep:
    """One traversal over a schema PK-FK link."""

    fk: ForeignKey
    direction: Direction

    @property
    def source_table(self) -> str:
        return self.fk.from_table if self.direction == "forward" else self.fk.to_table

    @property
    def target_table(self) -> str:
        return self.fk.to_table if self.direction == "forward" else self.fk.from_table

    @property
    def source_column(self) -> str:
        return self.fk.from_column if self.direction == "forward" else self.fk.to_column

    @property
    def target_column(self) -> str:
        return self.fk.to_column if self.direction == "forward" else self.fk.from_column

    @property
    def role(self) -> str:
        arrow = "->" if self.direction == "forward" else "<-"
        return f"{self.source_table}.{self.source_column}{arrow}{self.target_table}.{self.target_column}:{self.fk.edge_name}"


@dataclass(frozen=True)
class SchemaRoute:
    """A route from the task table to another table through schema links."""

    start_table: str
    steps: tuple[RouteStep, ...] = ()

    @property
    def end_table(self) -> str:
        if not self.steps:
            return self.start_table
        return self.steps[-1].target_table

    @property
    def hop_count(self) -> int:
        return len(self.steps)

    @property
    def table_path(self) -> tuple[str, ...]:
        tables = [self.start_table]
        tables.extend(step.target_table for step in self.steps)
        return tuple(tables)

    @property
    def name(self) -> str:
        if not self.steps:
            return self.start_table
        parts = [self.start_table]
        for step in self.steps:
            arrow = "->" if step.direction == "forward" else "<-"
            parts.append(f"{arrow}{step.target_table}[{step.fk.edge_name}]")
        return "".join(parts)

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(step.role for step in self.steps)

    @property
    def step_identities(self) -> tuple[tuple[str, str, str, str, str, str], ...]:
        return tuple(
            (
                step.fk.from_table,
                step.fk.from_column,
                step.fk.to_table,
                step.fk.to_column,
                step.fk.edge_name,
                step.direction,
            )
            for step in self.steps
        )

    def extend(self, step: RouteStep) -> "SchemaRoute":
        if self.end_table != step.source_table:
            raise ValueError("route extension must start at the current end table")
        return SchemaRoute(self.start_table, self.steps + (step,))


class RouteEnumerator:
    """Enumerate bounded schema routes from a target table."""

    def __init__(
        self,
        schema: DatabaseSchema,
        max_hops: int = 3,
        allow_cycles: bool = False,
        include_self: bool = True,
        skip_immediate_inverse: bool = True,
    ) -> None:
        if max_hops < 0:
            raise ValueError("max_hops must be non-negative")
        self.schema = schema
        self.max_hops = max_hops
        self.allow_cycles = allow_cycles
        self.include_self = include_self
        self.skip_immediate_inverse = skip_immediate_inverse

    def enumerate(self, start_table: str) -> tuple[SchemaRoute, ...]:
        self.schema.table(start_table)
        routes: list[SchemaRoute] = []
        frontier = [SchemaRoute(start_table)]
        if self.include_self:
            routes.append(frontier[0])

        for _ in range(self.max_hops):
            next_frontier: list[SchemaRoute] = []
            for route in frontier:
                for step in self._next_steps(route.end_table):
                    if self.skip_immediate_inverse and route.steps and _is_immediate_inverse(route.steps[-1], step):
                        continue
                    step_identity = (
                        step.fk.from_table,
                        step.fk.from_column,
                        step.fk.to_table,
                        step.fk.to_column,
                        step.fk.edge_name,
                        step.direction,
                    )
                    if not self.allow_cycles and step_identity in route.step_identities:
                        continue
                    new_route = route.extend(step)
                    routes.append(new_route)
                    next_frontier.append(new_route)
            frontier = next_frontier

        return tuple(routes)

    def _next_steps(self, table_name: str) -> tuple[RouteStep, ...]:
        steps: list[RouteStep] = []
        for fk in self.schema.incident_foreign_keys(table_name):
            if fk.from_table == table_name:
                steps.append(RouteStep(fk=fk, direction="forward"))
            if fk.to_table == table_name:
                steps.append(RouteStep(fk=fk, direction="backward"))
        return tuple(steps)


class AtomicRouteEnumerator(RouteEnumerator):
    """Enumerate bounded routes plus RelGNN-style atomic fact-table routes.

    For a multi-FK fact/bridge table ``F`` with links to ``A``, ``B`` and ``C``,
    starting at ``A`` this adds direct schema routes such as ``A <- F -> B`` and
    ``A <- F -> C``. These routes become coefficient/signal units rather than
    GNN message-passing edges.
    """

    def __init__(self, *args, atomic_only: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.atomic_only = atomic_only

    def enumerate(self, start_table: str) -> tuple[SchemaRoute, ...]:
        self.schema.table(start_table)
        if self.atomic_only:
            routes = [SchemaRoute(start_table)] if self.include_self else []
        else:
            routes = list(super().enumerate(start_table))
        seen = {route.name for route in routes}
        for fact_table, links in self._multi_fk_tables().items():
            start_links = [fk for fk in links if fk.to_table == start_table]
            if not start_links:
                continue
            for start_fk in start_links:
                first = RouteStep(fk=start_fk, direction="backward")
                for other_fk in links:
                    if other_fk == start_fk:
                        fact_route = SchemaRoute(start_table, (first,))
                        if fact_route.name not in seen and fact_route.hop_count <= self.max_hops:
                            seen.add(fact_route.name)
                            routes.append(fact_route)
                        continue
                    second = RouteStep(fk=other_fk, direction="forward")
                    atomic = SchemaRoute(start_table, (first, second))
                    if atomic.hop_count <= self.max_hops and atomic.name not in seen:
                        seen.add(atomic.name)
                        routes.append(atomic)
        return tuple(routes)

    def _multi_fk_tables(self) -> dict[str, tuple[ForeignKey, ...]]:
        grouped: dict[str, list[ForeignKey]] = {}
        for fk in self.schema.foreign_keys:
            grouped.setdefault(fk.from_table, []).append(fk)
        return {table: tuple(fks) for table, fks in grouped.items() if len(fks) >= 2}


def _is_immediate_inverse(prev: RouteStep, step: RouteStep) -> bool:
    return prev.fk == step.fk and prev.direction != step.direction
