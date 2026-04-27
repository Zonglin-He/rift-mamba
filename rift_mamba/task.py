"""Task specification and leakage-audit helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Iterable, Mapping, Sequence
import math

import numpy as np

from rift_mamba.schema import DatabaseSchema
from rift_mamba.records import RecordStore, TaskRow
from rift_mamba.routes import SchemaRoute


@dataclass(frozen=True)
class TaskSpec:
    target_table: str
    entity_column: str
    seed_time_column: str
    label_column: str
    task_type: str
    prediction_horizon: timedelta | None = None
    target_column: tuple[str, str] | None = None
    leakage_columns: tuple[tuple[str, str], ...] = ()
    src_entity_table: str | None = None
    src_entity_column: str | None = None
    dst_entity_table: str | None = None
    dst_entity_column: str | None = None
    eval_k: int | None = None


def build_exclude_columns(task: TaskSpec, schema: DatabaseSchema) -> tuple[tuple[str, str], ...]:
    """Columns that must not be used as input features for this task."""

    schema.table(task.target_table)
    excluded = set(task.leakage_columns)
    excluded.add((task.target_table, task.label_column))
    if task.target_column is not None:
        excluded.add(task.target_column)
    return tuple(sorted(excluded))


@dataclass(frozen=True)
class LeakageFinding:
    table: str
    column: str
    score: float
    reason: str
    route: str | None = None


def audit_proxy_leakage(
    rows: Sequence[Mapping[str, Any]],
    target_column: str,
    candidate_columns: Iterable[str] | None = None,
    threshold: float = 0.98,
) -> tuple[LeakageFinding, ...]:
    """Flag columns with near-deterministic association to the target.

    This is a train-split audit helper for autocomplete-style tasks. It is not
    a substitute for domain review, but it catches exact duplicates, nearly
    identical numeric columns, and one-to-one categorical proxies.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if not rows:
        return ()
    columns = tuple(candidate_columns or rows[0].keys())
    target = [row.get(target_column) for row in rows]
    findings: list[LeakageFinding] = []
    for column in columns:
        if column == target_column:
            continue
        values = [row.get(column) for row in rows]
        score, reason = association_score(values, target)
        if score >= threshold:
            findings.append(LeakageFinding(table="", column=column, score=score, reason=reason))
    return tuple(findings)


def audit_route_proxy_leakage(
    store: RecordStore,
    schema: DatabaseSchema,
    routes: Iterable[SchemaRoute],
    train_rows: Sequence[TaskRow],
    *,
    target_table: str,
    exclude_columns: Iterable[tuple[str, str]] = (),
    threshold: float = 0.98,
    max_pairs_per_column: int = 50_000,
) -> tuple[LeakageFinding, ...]:
    """Audit proxy leakage across rows reachable through schema routes.

    The target label is taken from ``TaskRow.label`` and paired with candidate
    feature values from every reachable path row. This catches target proxies
    outside the target table, such as one-hop fact columns or dimension fields
    that deterministically encode an autocomplete label.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    excluded = set(exclude_columns)
    values_by_column: dict[tuple[str, str, str], list[Any]] = {}
    targets_by_column: dict[tuple[str, str, str], list[Any]] = {}

    for task in train_rows:
        if task.label is None:
            continue
        for route in routes:
            for reachable in store.reachable(route, task, target_table=target_table):
                for table_name, row in reachable.path_rows:
                    table = schema.table(table_name)
                    for column in table.feature_columns:
                        if (table_name, column.name) in excluded:
                            continue
                        value = row.get(column.name)
                        if value is None:
                            continue
                        key = (route.name, table_name, column.name)
                        current = values_by_column.setdefault(key, [])
                        if len(current) >= max_pairs_per_column:
                            continue
                        current.append(value)
                        targets_by_column.setdefault(key, []).append(task.label)

    findings: list[LeakageFinding] = []
    seen_best: dict[tuple[str, str], LeakageFinding] = {}
    for (route_name, table_name, column_name), values in values_by_column.items():
        target = targets_by_column[(route_name, table_name, column_name)]
        score, reason = association_score(values, target)
        if score < threshold:
            continue
        finding = LeakageFinding(
            table=table_name,
            column=column_name,
            score=score,
            reason=reason,
            route=route_name,
        )
        key = (table_name, column_name)
        previous = seen_best.get(key)
        if previous is None or finding.score > previous.score:
            seen_best[key] = finding
    findings.extend(sorted(seen_best.values(), key=lambda item: (-item.score, item.table, item.column)))
    return tuple(findings)


def association_score(values: Sequence[Any], target: Sequence[Any]) -> tuple[float, str]:
    pairs = [(left, right) for left, right in zip(values, target, strict=False) if left is not None and right is not None]
    if len(pairs) < 2:
        return 0.0, "insufficient overlap"
    left = [item[0] for item in pairs]
    right = [item[1] for item in pairs]
    if left == right:
        return 1.0, "exact duplicate"
    if _all_numeric(left) and _all_numeric(right):
        x = np.asarray([float(v) for v in left], dtype=np.float64)
        y = np.asarray([float(v) for v in right], dtype=np.float64)
        if x.std() == 0 or y.std() == 0:
            return float(x.tolist() == y.tolist()), "constant numeric"
        return float(abs(np.corrcoef(x, y)[0, 1])), "absolute Pearson correlation"
    return _categorical_determinism(left, right), "categorical determinism"


def _all_numeric(values: Sequence[Any]) -> bool:
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        if math.isnan(number):
            return False
    return True


def _categorical_determinism(values: Sequence[Any], target: Sequence[Any]) -> float:
    buckets: dict[Any, dict[Any, int]] = {}
    for value, label in zip(values, target, strict=False):
        bucket = buckets.setdefault(value, {})
        bucket[label] = bucket.get(label, 0) + 1
    correct = sum(max(counts.values()) for counts in buckets.values())
    return correct / max(len(values), 1)
