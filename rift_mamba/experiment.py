"""Experiment assembly helpers for RIFT-Mamba."""

from __future__ import annotations

from dataclasses import dataclass

from rift_mamba.adapters import RelationalDatasetBundle
from rift_mamba.basis import BasisConfig, build_basis
from rift_mamba.coefficients import CoefficientExtractor, CoefficientMatrix
from rift_mamba.preprocessing import CoefficientStandardizer
from rift_mamba.records import TaskRow
from rift_mamba.routes import AtomicRouteEnumerator, RouteEnumerator
from rift_mamba.sequences import SequenceBatch, TemporalSequenceBuilder
from rift_mamba.task import LeakageFinding, audit_proxy_leakage, build_exclude_columns


@dataclass(frozen=True)
class PreparedExperiment:
    coefficients: CoefficientMatrix
    sequences: SequenceBatch
    standardizer: CoefficientStandardizer
    leakage_findings: tuple[LeakageFinding, ...] = ()


@dataclass(frozen=True)
class ExperimentConfig:
    max_hops: int = 3
    use_atomic_routes: bool = True
    sequence_max_len: int = 128
    basis_config: BasisConfig = BasisConfig()
    auto_leakage_audit: bool = True
    leakage_threshold: float = 0.98


def prepare_experiment(
    bundle: RelationalDatasetBundle,
    train_rows: tuple[TaskRow, ...],
    eval_rows: tuple[TaskRow, ...] = (),
    config: ExperimentConfig = ExperimentConfig(),
) -> tuple[PreparedExperiment, PreparedExperiment | None]:
    """Build train/eval coefficients and sequences with train-only normalization."""

    enumerator_cls = AtomicRouteEnumerator if config.use_atomic_routes else RouteEnumerator
    routes = enumerator_cls(bundle.schema, max_hops=config.max_hops).enumerate(bundle.task.target_table)
    exclude = set(build_exclude_columns(bundle.task, bundle.schema))
    leakage_findings: tuple[LeakageFinding, ...] = ()
    if config.auto_leakage_audit and bundle.task.label_column:
        train_ids = {row.entity_id for row in train_rows}
        target_table_rows = [
            row
            for row in bundle.tables.get(bundle.task.target_table, ())
            if row.get(bundle.task.entity_column) in train_ids
        ]
        leakage_findings = audit_proxy_leakage(
            target_table_rows,
            target_column=bundle.task.label_column,
            threshold=config.leakage_threshold,
        )
        exclude.update((bundle.task.target_table, finding.column) for finding in leakage_findings)
    bases = build_basis(bundle.schema, routes, config.basis_config, exclude_columns=exclude)
    store = bundle.record_store()
    extractor = CoefficientExtractor(store, bases)
    train_coeffs = extractor.transform(train_rows, target_table=bundle.task.target_table)
    standardizer = CoefficientStandardizer().fit(train_coeffs)
    train_coeffs = standardizer.transform(train_coeffs)
    temporal_routes = tuple(route for route in routes if route.hop_count > 0)
    sequence_builder = TemporalSequenceBuilder(bundle.schema, store, temporal_routes, max_len=config.sequence_max_len)
    train_sequences = sequence_builder.transform(train_rows, target_table=bundle.task.target_table)
    train = PreparedExperiment(train_coeffs, train_sequences, standardizer, leakage_findings)
    if not eval_rows:
        return train, None
    eval_coeffs = standardizer.transform(extractor.transform(eval_rows, target_table=bundle.task.target_table))
    eval_sequences = sequence_builder.transform(eval_rows, target_table=bundle.task.target_table)
    return train, PreparedExperiment(eval_coeffs, eval_sequences, standardizer, leakage_findings)
