"""Recommendation and link-prediction support for RIFT-Mamba."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import random

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from rift_mamba.adapters import RelationalDatasetBundle
from rift_mamba.basis import BasisConfig, build_basis
from rift_mamba.coefficients import CoefficientMatrix
from rift_mamba.nn import RiftMambaModel
from rift_mamba.preprocessing import CoefficientStandardizer
from rift_mamba.records import LinkTaskRow, TaskRow
from rift_mamba.routes import AtomicRouteEnumerator, RouteEnumerator
from rift_mamba.sequence_preprocessing import EventFeatureStandardizer
from rift_mamba.sequences import SequenceBatch, TemporalSequenceBuilder


@dataclass(frozen=True)
class PreparedLinkExperiment:
    src_coefficients: CoefficientMatrix
    dst_coefficients: CoefficientMatrix
    src_sequences: SequenceBatch
    dst_sequences: SequenceBatch
    link_rows: tuple[LinkTaskRow, ...]
    src_standardizer: CoefficientStandardizer
    dst_standardizer: CoefficientStandardizer
    src_event_standardizer: EventFeatureStandardizer
    dst_event_standardizer: EventFeatureStandardizer


@dataclass(frozen=True)
class LinkExperimentConfig:
    max_hops: int = 3
    use_atomic_routes: bool = True
    sequence_max_len: int = 128
    basis_config: BasisConfig = BasisConfig()
    atomic_only: bool = False


class LinkRiftDataset(Dataset):
    """Pair dataset over source/destination RIFT tensors."""

    def __init__(
        self,
        prepared: PreparedLinkExperiment,
        label_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.prepared = prepared
        self.labels = torch.as_tensor(
            np.asarray([row.label for row in prepared.link_rows]),
            dtype=label_dtype,
        )
        expected = len(prepared.link_rows)
        for name, matrix in (
            ("src_coefficients", prepared.src_coefficients),
            ("dst_coefficients", prepared.dst_coefficients),
        ):
            if matrix.values.shape[0] != expected:
                raise ValueError(f"{name} rows must match link rows")
        for name, sequences in (
            ("src_sequences", prepared.src_sequences),
            ("dst_sequences", prepared.dst_sequences),
        ):
            if sequences.values.shape[0] != expected:
                raise ValueError(f"{name} rows must match link rows")

    def __len__(self) -> int:
        return len(self.prepared.link_rows)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        p = self.prepared
        return {
            "src_alpha": torch.from_numpy(p.src_coefficients.values[index]).float(),
            "src_alpha_mask": torch.from_numpy(p.src_coefficients.masks[index]).bool(),
            "src_events": torch.from_numpy(p.src_sequences.values[index]).float(),
            "src_event_mask": torch.from_numpy(p.src_sequences.masks[index]).bool(),
            "dst_alpha": torch.from_numpy(p.dst_coefficients.values[index]).float(),
            "dst_alpha_mask": torch.from_numpy(p.dst_coefficients.masks[index]).bool(),
            "dst_events": torch.from_numpy(p.dst_sequences.values[index]).float(),
            "dst_event_mask": torch.from_numpy(p.dst_sequences.masks[index]).bool(),
            "label": self.labels[index],
        }


class PairRiftMambaModel(nn.Module):
    """Two-tower RIFT encoder with dot, bilinear, or MLP pair scoring."""

    def __init__(
        self,
        src_encoder: RiftMambaModel,
        dst_encoder: RiftMambaModel,
        scorer: str = "mlp",
        shared_dim: int | None = None,
    ) -> None:
        super().__init__()
        if scorer not in {"dot", "bilinear", "mlp"}:
            raise ValueError("scorer must be 'dot', 'bilinear' or 'mlp'")
        self.src_encoder = src_encoder
        self.dst_encoder = dst_encoder
        self.scorer = scorer
        dim = shared_dim or max(src_encoder.feature_dim, dst_encoder.feature_dim)
        self.src_proj = nn.Linear(src_encoder.feature_dim, dim)
        self.dst_proj = nn.Linear(dst_encoder.feature_dim, dim)
        if scorer == "bilinear":
            self.bilinear = nn.Bilinear(dim, dim, 1)
        elif scorer == "mlp":
            self.mlp = nn.Sequential(
                nn.LayerNorm(dim * 4),
                nn.Linear(dim * 4, dim),
                nn.GELU(),
                nn.Linear(dim, 1),
            )

    def forward(
        self,
        src_alpha: Tensor,
        src_alpha_mask: Tensor,
        dst_alpha: Tensor,
        dst_alpha_mask: Tensor,
        src_events: Tensor | None = None,
        src_event_mask: Tensor | None = None,
        dst_events: Tensor | None = None,
        dst_event_mask: Tensor | None = None,
    ) -> Tensor:
        src = self.src_proj(self.src_encoder.encode(src_alpha, src_alpha_mask, src_events, src_event_mask))
        dst = self.dst_proj(self.dst_encoder.encode(dst_alpha, dst_alpha_mask, dst_events, dst_event_mask))
        if self.scorer == "dot":
            return torch.sum(src * dst, dim=-1)
        if self.scorer == "bilinear":
            return self.bilinear(src, dst).squeeze(-1)
        features = torch.cat([src, dst, src * dst, torch.abs(src - dst)], dim=-1)
        return self.mlp(features).squeeze(-1)


def prepare_link_experiment(
    bundle: RelationalDatasetBundle,
    link_rows: Sequence[LinkTaskRow],
    config: LinkExperimentConfig = LinkExperimentConfig(),
) -> PreparedLinkExperiment:
    if bundle.task.src_entity_table is None or bundle.task.dst_entity_table is None:
        raise ValueError("bundle.task must define src/dst entity tables for link prediction")
    rows = tuple(link_rows)
    enumerator_cls = AtomicRouteEnumerator if config.use_atomic_routes else RouteEnumerator
    if config.use_atomic_routes:
        src_routes = enumerator_cls(
            bundle.schema,
            max_hops=config.max_hops,
            atomic_only=config.atomic_only,
        ).enumerate(bundle.task.src_entity_table)
        dst_routes = enumerator_cls(
            bundle.schema,
            max_hops=config.max_hops,
            atomic_only=config.atomic_only,
        ).enumerate(bundle.task.dst_entity_table)
    else:
        src_routes = enumerator_cls(bundle.schema, max_hops=config.max_hops).enumerate(bundle.task.src_entity_table)
        dst_routes = enumerator_cls(bundle.schema, max_hops=config.max_hops).enumerate(bundle.task.dst_entity_table)
    src_bases = build_basis(bundle.schema, src_routes, config.basis_config)
    dst_bases = build_basis(bundle.schema, dst_routes, config.basis_config)
    backend = bundle.coefficient_backend()
    store = backend.record_store()
    src_tasks = tuple(TaskRow(row.row_id, row.src_id, row.seed_time, row.label) for row in rows)
    dst_tasks = tuple(TaskRow(row.row_id, row.dst_id, row.seed_time, row.label) for row in rows)
    src_coeffs = backend.coefficient_extractor(src_bases).transform(src_tasks, bundle.task.src_entity_table)
    dst_coeffs = backend.coefficient_extractor(dst_bases).transform(dst_tasks, bundle.task.dst_entity_table)
    src_standardizer = CoefficientStandardizer().fit(src_coeffs)
    dst_standardizer = CoefficientStandardizer().fit(dst_coeffs)
    src_coeffs = src_standardizer.transform(src_coeffs)
    dst_coeffs = dst_standardizer.transform(dst_coeffs)
    src_temporal_routes = tuple(route for route in src_routes if route.hop_count > 0)
    dst_temporal_routes = tuple(route for route in dst_routes if route.hop_count > 0)
    src_builder = TemporalSequenceBuilder(bundle.schema, store, src_temporal_routes, max_len=config.sequence_max_len)
    dst_builder = TemporalSequenceBuilder(bundle.schema, store, dst_temporal_routes, max_len=config.sequence_max_len)
    src_sequences = src_builder.transform(src_tasks, bundle.task.src_entity_table)
    dst_sequences = dst_builder.transform(dst_tasks, bundle.task.dst_entity_table)
    src_event_standardizer = EventFeatureStandardizer().fit(src_sequences)
    dst_event_standardizer = EventFeatureStandardizer().fit(dst_sequences)
    src_sequences = src_event_standardizer.transform(src_sequences)
    dst_sequences = dst_event_standardizer.transform(dst_sequences)
    return PreparedLinkExperiment(
        src_coefficients=src_coeffs,
        dst_coefficients=dst_coeffs,
        src_sequences=src_sequences,
        dst_sequences=dst_sequences,
        link_rows=rows,
        src_standardizer=src_standardizer,
        dst_standardizer=dst_standardizer,
        src_event_standardizer=src_event_standardizer,
        dst_event_standardizer=dst_event_standardizer,
    )


def transform_link_experiment(
    bundle: RelationalDatasetBundle,
    link_rows: Sequence[LinkTaskRow],
    fitted: PreparedLinkExperiment,
) -> PreparedLinkExperiment:
    """Transform eval/test link rows with train-fitted bases and scalers."""

    if bundle.task.src_entity_table is None or bundle.task.dst_entity_table is None:
        raise ValueError("bundle.task must define src/dst entity tables for link prediction")
    rows = tuple(link_rows)
    backend = bundle.coefficient_backend()
    store = backend.record_store()
    src_tasks = tuple(TaskRow(row.row_id, row.src_id, row.seed_time, row.label) for row in rows)
    dst_tasks = tuple(TaskRow(row.row_id, row.dst_id, row.seed_time, row.label) for row in rows)
    src_coeffs = backend.coefficient_extractor(fitted.src_coefficients.bases).transform(
        src_tasks,
        bundle.task.src_entity_table,
    )
    dst_coeffs = backend.coefficient_extractor(fitted.dst_coefficients.bases).transform(
        dst_tasks,
        bundle.task.dst_entity_table,
    )
    src_coeffs = fitted.src_standardizer.transform(src_coeffs)
    dst_coeffs = fitted.dst_standardizer.transform(dst_coeffs)
    src_builder = TemporalSequenceBuilder(
        bundle.schema,
        store,
        fitted.src_sequences.routes,
        max_len=fitted.src_sequences.values.shape[2],
        event_columns=fitted.src_sequences.event_columns,
    )
    dst_builder = TemporalSequenceBuilder(
        bundle.schema,
        store,
        fitted.dst_sequences.routes,
        max_len=fitted.dst_sequences.values.shape[2],
        event_columns=fitted.dst_sequences.event_columns,
    )
    src_sequences = fitted.src_event_standardizer.transform(
        src_builder.transform(src_tasks, bundle.task.src_entity_table)
    )
    dst_sequences = fitted.dst_event_standardizer.transform(
        dst_builder.transform(dst_tasks, bundle.task.dst_entity_table)
    )
    return PreparedLinkExperiment(
        src_coefficients=src_coeffs,
        dst_coefficients=dst_coeffs,
        src_sequences=src_sequences,
        dst_sequences=dst_sequences,
        link_rows=rows,
        src_standardizer=fitted.src_standardizer,
        dst_standardizer=fitted.dst_standardizer,
        src_event_standardizer=fitted.src_event_standardizer,
        dst_event_standardizer=fitted.dst_event_standardizer,
    )


def sample_negative_links(
    positive_rows: Sequence[LinkTaskRow],
    candidate_dst_ids: Sequence[object],
    num_negatives: int = 1,
    seed: int = 0,
) -> tuple[LinkTaskRow, ...]:
    """Add uniformly sampled negative dst entities per positive link."""

    if num_negatives < 0:
        raise ValueError("num_negatives must be non-negative")
    candidates = tuple(candidate_dst_ids)
    if not candidates and num_negatives:
        raise ValueError("candidate_dst_ids cannot be empty when num_negatives > 0")
    rng = random.Random(seed)
    positives = tuple(positive_rows)
    positive_by_src: dict[object, set[object]] = {}
    for row in positives:
        positive_by_src.setdefault(row.src_id, set()).add(row.dst_id)
    output: list[LinkTaskRow] = list(positives)
    for row in positives:
        banned = positive_by_src.get(row.src_id, set())
        available = [dst for dst in candidates if dst not in banned]
        if not available:
            continue
        for index in range(num_negatives):
            output.append(
                LinkTaskRow(
                    row_id=f"{row.row_id}:neg:{index}",
                    src_id=row.src_id,
                    dst_id=rng.choice(available),
                    seed_time=row.seed_time,
                    label=0,
                )
            )
    return tuple(output)
