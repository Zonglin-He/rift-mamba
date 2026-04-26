"""Split helpers for temporal relational tasks."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

from rift_mamba.records import TaskRow
from rift_mamba.time import parse_time


def temporal_split(
    rows: Iterable[TaskRow],
    val_time: datetime | str,
    test_time: datetime | str,
) -> tuple[tuple[TaskRow, ...], tuple[TaskRow, ...], tuple[TaskRow, ...]]:
    val_cutoff = parse_time(val_time)
    test_cutoff = parse_time(test_time)
    if val_cutoff is None or test_cutoff is None:
        raise ValueError("val_time and test_time are required")
    if test_cutoff <= val_cutoff:
        raise ValueError("test_time must be later than val_time")
    train: list[TaskRow] = []
    val: list[TaskRow] = []
    test: list[TaskRow] = []
    for row in rows:
        cutoff = row.cutoff
        if cutoff < val_cutoff:
            train.append(row)
        elif cutoff < test_cutoff:
            val.append(row)
        else:
            test.append(row)
    return tuple(train), tuple(val), tuple(test)
