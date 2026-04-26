"""RIFT-Mamba reference implementation.

The package implements a leakage-safe relational signal pipeline:

database records -> schema routes -> causal reachable rows -> relational
coefficients -> learnable basis synthesis -> route-wise sequence encoder.
"""

from rift_mamba.basis import BasisConfig, RelationalBasis, build_basis
from rift_mamba.coefficients import CoefficientExtractor, CoefficientMatrix
from rift_mamba.dataset import RiftDataset
from rift_mamba.nn import RiftMambaModel
from rift_mamba.preprocessing import CoefficientStandardizer
from rift_mamba.records import RecordStore, TaskRow
from rift_mamba.routes import RouteEnumerator, RouteStep, SchemaRoute
from rift_mamba.schema import ColumnSchema, DatabaseSchema, ForeignKey, TableSchema
from rift_mamba.sequences import SequenceBatch, TemporalSequenceBuilder

__all__ = [
    "BasisConfig",
    "CoefficientExtractor",
    "CoefficientMatrix",
    "CoefficientStandardizer",
    "ColumnSchema",
    "DatabaseSchema",
    "ForeignKey",
    "RecordStore",
    "RelationalBasis",
    "RiftDataset",
    "RiftMambaModel",
    "RouteEnumerator",
    "RouteStep",
    "SchemaRoute",
    "SequenceBatch",
    "TableSchema",
    "TaskRow",
    "TemporalSequenceBuilder",
    "build_basis",
]
