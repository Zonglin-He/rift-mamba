"""RelBench v2 loader boundary including autocomplete task specs."""

from __future__ import annotations

from rift_mamba.adapters import RelBenchAdapter, RelationalDatasetBundle


def load_relbench_v2(source=None, **kwargs) -> RelationalDatasetBundle:
    if source is None:
        source = kwargs
    return RelBenchAdapter.from_relbench(source).load()
