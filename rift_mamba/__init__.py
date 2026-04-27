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
from rift_mamba.metrics import (
    accuracy_score,
    average_precision_score,
    mae,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
    roc_auc_score,
)
from rift_mamba.nn import (
    BASIS_MODES,
    BasisGraphEncoder,
    BasisMixerEncoder,
    BiMambaBasisEncoder,
    DeepSetBasisEncoder,
    FTTransformerBasisEncoder,
    MultiScanMambaBasisEncoder,
    PerceiverBasisEncoder,
    RelationalBasisAttentionEncoder,
    RelationalCNNEncoder,
    RiftMambaModel,
    RouteMambaBasisEncoder,
    RouteSetEncoder,
    SetTransformerBasisEncoder,
)
from rift_mamba.preprocessing import CoefficientStandardizer
from rift_mamba.pretraining import TaskVectorHead, TaskVectorTargetBuilder, TaskVectorTargets, task_vector_cosine_loss
from rift_mamba.recommendation import (
    LinkExperimentConfig,
    LinkRiftDataset,
    PairRiftMambaModel,
    PreparedLinkExperiment,
    prepare_link_experiment,
    sample_negative_links,
    transform_link_experiment,
)
from rift_mamba.records import LinkTaskRow, RecordStore, TaskRow
from rift_mamba.routes import AtomicRouteEnumerator, RouteEnumerator, RouteStep, SchemaRoute
from rift_mamba.schema import ColumnSchema, DatabaseSchema, ForeignKey, TableSchema
from rift_mamba.semantic import HashTextEncoder, SchemaSemanticEncoder, SentenceTransformerTextEncoder
from rift_mamba.sequence_preprocessing import EventFeatureStandardizer
from rift_mamba.sequences import SequenceBatch, TemporalSequenceBuilder
from rift_mamba.splits import temporal_split
from rift_mamba.task import LeakageFinding, TaskSpec, audit_proxy_leakage, audit_route_proxy_leakage, build_exclude_columns
from rift_mamba.trainer import EpochMetrics, evaluate_epoch, train_epoch

__all__ = [
    "AtomicRouteEnumerator",
    "BASIS_MODES",
    "BasisConfig",
    "BasisGraphEncoder",
    "BasisLayout",
    "BasisMixerEncoder",
    "BiMambaBasisEncoder",
    "CoefficientExtractor",
    "CoefficientMatrix",
    "CoefficientStandardizer",
    "ColumnSchema",
    "CompositeBasisSpec",
    "CompositeRelationalBasis",
    "DatabaseSchema",
    "DFSMLPBaseline",
    "DeepSetBasisEncoder",
    "DuckDBBackend",
    "ExperimentConfig",
    "EventFeatureStandardizer",
    "EpochMetrics",
    "ExternalBaseline",
    "ForeignKey",
    "FTTransformerBasisEncoder",
    "HashTextEncoder",
    "InMemoryBackend",
    "LeakageFinding",
    "LinkExperimentConfig",
    "LinkRiftDataset",
    "LinkTaskRow",
    "MultiScanMambaBasisEncoder",
    "PairRiftMambaModel",
    "PerceiverBasisEncoder",
    "PreparedExperiment",
    "PreparedLinkExperiment",
    "PolarsBackend",
    "RecordStore",
    "RelationalBasis",
    "RelationalBasisAttentionEncoder",
    "RelationalCNNEncoder",
    "RelationalDatasetBundle",
    "RelBenchAdapter",
    "RiftDataset",
    "RiftMambaModel",
    "RouteEnumerator",
    "RouteMambaBasisEncoder",
    "RouteStep",
    "RouteSetEncoder",
    "SchemaRoute",
    "SchemaSemanticEncoder",
    "SentenceTransformerTextEncoder",
    "SequenceBatch",
    "SetTransformerBasisEncoder",
    "TableSchema",
    "TaskRow",
    "TaskSpec",
    "TaskVectorHead",
    "TaskVectorTargetBuilder",
    "TaskVectorTargets",
    "audit_proxy_leakage",
    "audit_route_proxy_leakage",
    "TemporalSequenceBuilder",
    "accuracy_score",
    "average_precision_score",
    "build_basis",
    "build_composite_basis",
    "build_exclude_columns",
    "default_baseline_registry",
    "make_backend",
    "mae",
    "mrr_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "prepare_experiment",
    "prepare_link_experiment",
    "recall_at_k",
    "rmse",
    "roc_auc_score",
    "sample_negative_links",
    "task_vector_cosine_loss",
    "temporal_split",
    "evaluate_epoch",
    "train_epoch",
    "transform_link_experiment",
]
