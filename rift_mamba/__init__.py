"""RIFT-Mamba reference implementation.

The package implements a leakage-safe relational signal pipeline:

database records -> schema routes -> causal reachable rows -> relational
coefficients -> learnable basis synthesis -> route-wise sequence encoder.
"""

from rift_mamba.adapters import RelBenchAdapter, RelationalDatasetBundle
from rift_mamba.backends import DuckDBBackend, InMemoryBackend, PolarsBackend, make_backend
from rift_mamba.baselines import DFSMLPBaseline, ExternalBaseline, default_baseline_registry
from rift_mamba.basis import (
    BasisConfig,
    CompositeBasisSpec,
    CompositeRelationalBasis,
    RelationalBasis,
    build_basis,
    build_composite_basis,
)
from rift_mamba.coefficients import CoefficientExtractor, CoefficientMatrix
from rift_mamba.dataset import RiftDataset
from rift_mamba.experiment import ExperimentConfig, PreparedExperiment, prepare_experiment
from rift_mamba.layout import BasisLayout
from rift_mamba.metrics import accuracy_score, mae, rmse
from rift_mamba.nn import RelationalCNNEncoder, RiftMambaModel
from rift_mamba.preprocessing import CoefficientStandardizer
from rift_mamba.pretraining import TaskVectorHead, TaskVectorTargetBuilder, TaskVectorTargets, task_vector_cosine_loss
from rift_mamba.records import RecordStore, TaskRow
from rift_mamba.routes import AtomicRouteEnumerator, RouteEnumerator, RouteStep, SchemaRoute
from rift_mamba.schema import ColumnSchema, DatabaseSchema, ForeignKey, TableSchema
from rift_mamba.semantic import HashTextEncoder, SchemaSemanticEncoder, SentenceTransformerTextEncoder
from rift_mamba.sequences import SequenceBatch, TemporalSequenceBuilder
from rift_mamba.splits import temporal_split
from rift_mamba.task import LeakageFinding, TaskSpec, audit_proxy_leakage, build_exclude_columns

__all__ = [
    "AtomicRouteEnumerator",
    "BasisConfig",
    "BasisLayout",
    "CoefficientExtractor",
    "CoefficientMatrix",
    "CoefficientStandardizer",
    "ColumnSchema",
    "CompositeBasisSpec",
    "CompositeRelationalBasis",
    "DatabaseSchema",
    "DFSMLPBaseline",
    "DuckDBBackend",
    "ExperimentConfig",
    "ExternalBaseline",
    "ForeignKey",
    "HashTextEncoder",
    "InMemoryBackend",
    "LeakageFinding",
    "PreparedExperiment",
    "PolarsBackend",
    "RecordStore",
    "RelationalBasis",
    "RelationalCNNEncoder",
    "RelationalDatasetBundle",
    "RelBenchAdapter",
    "RiftDataset",
    "RiftMambaModel",
    "RouteEnumerator",
    "RouteStep",
    "SchemaRoute",
    "SchemaSemanticEncoder",
    "SentenceTransformerTextEncoder",
    "SequenceBatch",
    "TableSchema",
    "TaskRow",
    "TaskSpec",
    "TaskVectorHead",
    "TaskVectorTargetBuilder",
    "TaskVectorTargets",
    "audit_proxy_leakage",
    "TemporalSequenceBuilder",
    "accuracy_score",
    "build_basis",
    "build_composite_basis",
    "build_exclude_columns",
    "default_baseline_registry",
    "make_backend",
    "mae",
    "prepare_experiment",
    "rmse",
    "task_vector_cosine_loss",
    "temporal_split",
]
