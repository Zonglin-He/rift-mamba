"""RelBench v2 loader boundary including autocomplete task specs."""

from __future__ import annotations

from rift_mamba.adapters import RelBenchAdapter, RelationalDatasetBundle


def load_relbench_v2(
    dataset_name: str | None = None,
    task_name: str | None = None,
    *,
    split: str | None = None,
    download: bool = True,
    backend: str = "materialized",
    include_test_labels: bool = False,
    source=None,
    **kwargs,
) -> RelationalDatasetBundle:
    """Load a RelBench v2 dataset/task, including autocomplete metadata."""

    if source is None and task_name is None and dataset_name is not None and not isinstance(dataset_name, str):
        source = dataset_name
    if source is not None or dataset_name is None or task_name is None:
        source = source if source is not None else kwargs
        return RelBenchAdapter.from_relbench(source).load()
    return RelBenchAdapter.from_official(
        dataset_name,
        task_name,
        split=split,
        download=download,
        backend=backend,
        include_test_labels=include_test_labels,
    ).load()
