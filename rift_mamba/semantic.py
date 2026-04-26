"""Schema and value semantic encoders.

The default encoder is deterministic and dependency-free. Production systems
can inject a frozen LLM/text encoder by implementing the same ``encode`` method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol
import hashlib

import numpy as np

from rift_mamba.basis import RelationalBasis
from rift_mamba.routes import SchemaRoute
from rift_mamba.schema import ColumnSchema, ForeignKey


class TextEncoder(Protocol):
    dim: int

    def encode(self, text: str) -> np.ndarray:
        ...


@dataclass(frozen=True)
class HashTextEncoder:
    """A stable signed hashing encoder for metadata and category values."""

    dim: int = 32
    salt: str = "rift-mamba"

    def encode(self, text: str) -> np.ndarray:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = _tokenize(text)
        if not tokens:
            tokens = [""]
        for token in tokens:
            digest = hashlib.blake2b(f"{self.salt}:{token}".encode("utf-8"), digest_size=16).digest()
            bucket = int.from_bytes(digest[:8], byteorder="big", signed=False) % self.dim
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector


@dataclass
class SentenceTransformerTextEncoder:
    """Optional frozen sentence-transformers encoder for schema semantics."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize: bool = True
    _model: object | None = field(default=None, init=False, repr=False)
    _dim: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:
            raise ImportError(
                "SentenceTransformerTextEncoder requires the optional 'sentence-transformers' package. "
                "Install rift-mamba[text] or inject another TextEncoder."
            ) from exc
        self._model = SentenceTransformer(self.model_name)
        dim = self._model.get_sentence_embedding_dimension()
        self._dim = int(dim) if dim is not None else int(self.encode("dimension probe").shape[0])

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("SentenceTransformerTextEncoder is not initialized")
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("SentenceTransformerTextEncoder is not initialized")
        vector = self._model.encode(text, normalize_embeddings=self.normalize)
        return np.asarray(vector, dtype=np.float32)


class SchemaSemanticEncoder:
    """Build frozen semantic vectors for schema objects and basis tokens."""

    def __init__(self, text_encoder: TextEncoder | None = None) -> None:
        self.text_encoder = text_encoder or HashTextEncoder()

    @property
    def dim(self) -> int:
        return self.text_encoder.dim

    def encode_table(self, table: str) -> np.ndarray:
        return self.text_encoder.encode(f"table {table}")

    def encode_column(self, table: str, column: ColumnSchema | str, semantic_name: str | None = None) -> np.ndarray:
        if isinstance(column, ColumnSchema):
            name = column.name
            kind = column.kind
            semantic = semantic_name or column.semantic_name
        else:
            name = column
            kind = "unknown"
            semantic = semantic_name
        text = semantic or f"{kind} column {name} of table {table}"
        return self.text_encoder.encode(text)

    def encode_fk_role(self, fk: ForeignKey) -> np.ndarray:
        role = fk.role or fk.edge_name
        text = f"foreign key role {role}: {fk.from_table}.{fk.from_column} references {fk.to_table}.{fk.to_column}"
        return self.text_encoder.encode(text)

    def encode_route(self, route: SchemaRoute) -> np.ndarray:
        role_text = " ".join(route.roles)
        return self.text_encoder.encode(f"schema route {route.name} tables {' '.join(route.table_path)} roles {role_text}")

    def encode_category(self, table: str, column: str, value: object) -> np.ndarray:
        return self.text_encoder.encode(f"value {value} in column {column} of table {table}")

    def encode_basis(self, basis: RelationalBasis) -> np.ndarray:
        window = "all history" if basis.window is None else f"{basis.window.total_seconds() / 86400:g} day window"
        column = basis.column_name or "row count"
        role_text = " ".join(basis.route.roles)
        text = (
            f"basis route {basis.route.name}; end table {basis.end_table}; column {column}; "
            f"kind {basis.column_kind}; aggregation {basis.aggregator}; window {window}; roles {role_text}"
        )
        return self.text_encoder.encode(text)

    def basis_matrix(self, bases: tuple[RelationalBasis, ...]) -> np.ndarray:
        if not bases:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.encode_basis(basis) for basis in bases]).astype(np.float32)


def _tokenize(text: str) -> list[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [part for part in cleaned.split() if part]
