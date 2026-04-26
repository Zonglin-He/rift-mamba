"""RelBench v1 loader boundary.

This module intentionally keeps release-specific object parsing outside the
core model. It accepts materialized RelBench-like objects directly and provides
a named entry point for project-specific official RelBench v1 adapters.
"""

from __future__ import annotations

from rift_mamba.adapters import RelBenchAdapter, RelationalDatasetBundle


def load_relbench_v1(source=None, **kwargs) -> RelationalDatasetBundle:
    if source is None:
        source = kwargs
    return RelBenchAdapter.from_relbench(source).load()
