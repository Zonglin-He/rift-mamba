"""Neural modules for RIFT-Mamba."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from rift_mamba.basis import RelationalBasis
from rift_mamba.layout import BasisLayout
from rift_mamba.semantic import SchemaSemanticEncoder


class RelationalBasisSynthesizer(nn.Module):
    """Compose alpha_b(q) with learnable basis and missing-value tensors.

    For each basis token:

        e_b(q) = m_b alpha_b(q) Psi_b + (1 - m_b) Omega_b

    The output is a dense sequence of basis-indexed neural tokens.
    """

    def __init__(
        self,
        bases: Iterable[RelationalBasis],
        d_model: int,
        semantic_encoder: SchemaSemanticEncoder | None = None,
        semantic_matrix: np.ndarray | Tensor | None = None,
        value_embedding_buckets: int = 256,
    ) -> None:
        super().__init__()
        self.bases = tuple(bases)
        self.num_bases = len(self.bases)
        self.d_model = d_model
        self.basis = nn.Parameter(torch.empty(self.num_bases, d_model))
        self.missing_basis = nn.Parameter(torch.empty(self.num_bases, d_model))
        self.alpha_scale = nn.Parameter(torch.ones(self.num_bases))
        self.alpha_bias = nn.Parameter(torch.zeros(self.num_bases))
        self.mask_embedding = nn.Embedding(2, d_model)
        if value_embedding_buckets <= 0:
            raise ValueError("value_embedding_buckets must be positive")
        self.value_embedding_buckets = value_embedding_buckets
        self.value_embedding = nn.Embedding(self.num_bases * value_embedding_buckets, d_model)
        categorical_mask = torch.tensor(
            [basis.column_kind in {"categorical", "text"} for basis in self.bases],
            dtype=torch.bool,
        )
        self.register_buffer("categorical_value_mask", categorical_mask)
        if semantic_matrix is None:
            encoder = semantic_encoder or SchemaSemanticEncoder()
            semantic_matrix = encoder.basis_matrix(self.bases)
        semantic_tensor = torch.as_tensor(semantic_matrix, dtype=torch.float32)
        if semantic_tensor.ndim != 2 or semantic_tensor.shape[0] != self.num_bases:
            raise ValueError("semantic_matrix must have shape [num_bases, semantic_dim]")
        self.register_buffer("semantic_matrix", semantic_tensor)
        self.semantic_proj = nn.Linear(semantic_tensor.shape[1], d_model, bias=False)
        nn.init.normal_(self.basis, mean=0.0, std=0.02)
        nn.init.normal_(self.missing_basis, mean=0.0, std=0.02)

    def forward(self, alpha: Tensor, mask: Tensor) -> Tensor:
        if alpha.ndim != 2 or mask.ndim != 2:
            raise ValueError("alpha and mask must have shape [batch, num_bases]")
        if alpha.shape != mask.shape or alpha.shape[1] != self.num_bases:
            raise ValueError("alpha/mask shape does not match basis count")
        mask_f = mask.to(dtype=alpha.dtype)
        scaled_alpha = alpha * self.alpha_scale.unsqueeze(0) + self.alpha_bias.unsqueeze(0)
        present_tokens = scaled_alpha.unsqueeze(-1) * self.basis.unsqueeze(0)
        missing_tokens = self.missing_basis.unsqueeze(0).expand(alpha.shape[0], -1, -1)
        tokens = mask_f.unsqueeze(-1) * present_tokens + (1.0 - mask_f).unsqueeze(-1) * missing_tokens
        tokens = tokens + self.mask_embedding(mask.long())
        tokens = tokens + self.semantic_proj(self.semantic_matrix).unsqueeze(0)
        if bool(self.categorical_value_mask.any()):
            bucket_ids = torch.clamp(
                torch.round((torch.clamp(alpha, -1.0, 1.0) + 1.0) * 0.5 * (self.value_embedding_buckets - 1)),
                min=0,
                max=self.value_embedding_buckets - 1,
            ).long()
            basis_offsets = torch.arange(self.num_bases, device=alpha.device).view(1, -1) * self.value_embedding_buckets
            value_tokens = self.value_embedding(bucket_ids + basis_offsets)
            cat_mask = self.categorical_value_mask.to(device=alpha.device).view(1, -1, 1)
            tokens = tokens + value_tokens * cat_mask * mask_f.unsqueeze(-1)
        return tokens

    def sum_signal(self, alpha: Tensor, mask: Tensor) -> Tensor:
        """Literal relational inverse transform: sum_b e_b(q)."""

        return self.forward(alpha, mask).sum(dim=1)


class CausalGatedSSMBlock(nn.Module):
    """A compact causal selective-state block used when mamba-ssm is absent."""

    def __init__(self, d_model: int, expansion: int = 2, kernel_size: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        inner = d_model * expansion
        self.in_proj = nn.Linear(d_model, inner * 2)
        self.conv = nn.Conv1d(inner, inner, kernel_size=kernel_size, groups=inner, padding=kernel_size - 1)
        self.a_proj = nn.Linear(inner, inner)
        self.b_proj = nn.Linear(inner, inner)
        self.c_proj = nn.Linear(inner, inner)
        self.out_proj = nn.Linear(inner, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.kernel_size = kernel_size

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        residual = x
        x = self.norm(x)
        projected, gate = self.in_proj(x).chunk(2, dim=-1)
        if mask is not None:
            mask_f = mask.to(dtype=projected.dtype).unsqueeze(-1)
            projected = projected * mask_f
            gate = gate * mask_f
        conv_in = projected.transpose(1, 2)
        conv_out = self.conv(conv_in)
        conv_out = conv_out[:, :, : projected.shape[1]].transpose(1, 2)
        u = F.silu(conv_out) * torch.sigmoid(gate)

        a = torch.sigmoid(self.a_proj(u))
        b = torch.sigmoid(self.b_proj(u))
        c = torch.tanh(self.c_proj(u))
        state = torch.zeros(u.shape[0], u.shape[2], device=u.device, dtype=u.dtype)
        outputs: list[Tensor] = []
        if mask is None:
            step_mask = None
        else:
            step_mask = mask.to(dtype=u.dtype)

        for index in range(u.shape[1]):
            state = a[:, index] * state + b[:, index] * u[:, index]
            out = c[:, index] * state + u[:, index]
            if step_mask is not None:
                out = out * step_mask[:, index].unsqueeze(-1)
                state = state * step_mask[:, index].unsqueeze(-1)
            outputs.append(out)

        y = torch.stack(outputs, dim=1)
        return residual + self.dropout(self.out_proj(y))


class MambaBlock(nn.Module):
    """Use mamba-ssm when installed, otherwise a causal gated SSM fallback."""

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
        use_mamba_ssm: bool = True,
        allow_masked_mamba: bool = True,
    ) -> None:
        super().__init__()
        self.allow_masked_mamba = allow_masked_mamba
        self.fallback = CausalGatedSSMBlock(d_model=d_model, dropout=dropout)
        if not use_mamba_ssm:
            self.impl: nn.Module = self.fallback
            self.uses_mamba_ssm = False
        else:
            try:
                from mamba_ssm import Mamba  # type: ignore
            except Exception:
                self.impl = self.fallback
                self.uses_mamba_ssm = False
            else:
                self.impl = Mamba(d_model=d_model)
                self.dropout = nn.Dropout(dropout)
                self.uses_mamba_ssm = True

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        if mask is not None and not bool(mask.all()) and not self.allow_masked_mamba:
            return self.fallback(x, mask)
        if self.uses_mamba_ssm:
            if mask is not None:
                x = x * mask.to(dtype=x.dtype).unsqueeze(-1)
            y = self.impl(x)
            if mask is not None:
                y = y * mask.to(dtype=y.dtype).unsqueeze(-1)
            return self.dropout(y)  # type: ignore[attr-defined]
        return self.impl(x, mask)

    @property
    def implementation_name(self) -> str:
        if self.uses_mamba_ssm:
            return "mamba_ssm"
        return "fallback_causal_gated_ssm"

    def forward_exact_mask(self, x: Tensor, mask: Tensor) -> Tensor:
        """Run the exact-mask fallback for diagnostics."""

        return self.fallback(x, mask)

    def masked_approximation_error(self, x: Tensor, mask: Tensor) -> Tensor:
        """Compare current masked behavior against the exact-mask fallback."""

        with torch.no_grad():
            exact = self.forward_exact_mask(x, mask)
            approx = self.forward(x, mask)
            return (exact - approx).abs().max()


class StackedMambaEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_mamba_ssm: bool = True,
        allow_masked_mamba: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    dropout=dropout,
                    use_mamba_ssm=use_mamba_ssm,
                    allow_masked_mamba=allow_masked_mamba,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class RouteWiseSequenceEncoder(nn.Module):
    """Encode S_r(q) for each temporal route and pool routes with learned gates."""

    def __init__(
        self,
        num_routes: int,
        event_dim: int,
        d_model: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_mamba_ssm: bool = True,
        allow_masked_mamba: bool = True,
    ) -> None:
        super().__init__()
        self.num_routes = num_routes
        self.event_proj = nn.Linear(event_dim, d_model)
        self.route_embedding = nn.Embedding(num_routes, d_model)
        self.encoder = StackedMambaEncoder(
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
            use_mamba_ssm=use_mamba_ssm,
            allow_masked_mamba=allow_masked_mamba,
        )
        self.route_gate = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, events: Tensor, event_mask: Tensor) -> Tensor:
        if events.ndim != 4:
            raise ValueError("events must have shape [batch, routes, length, event_dim]")
        batch, routes, length, _ = events.shape
        if routes != self.num_routes:
            raise ValueError("events route dimension does not match num_routes")
        route_ids = torch.arange(routes, device=events.device)
        route_emb = self.route_embedding(route_ids).view(1, routes, 1, -1)
        tokens = self.event_proj(events) + route_emb
        tokens = tokens * event_mask.to(dtype=tokens.dtype).unsqueeze(-1)
        flat_tokens = tokens.reshape(batch * routes, length, -1)
        flat_mask = event_mask.reshape(batch * routes, length)
        encoded = self.encoder(flat_tokens, flat_mask)
        route_repr = masked_last(encoded, flat_mask).reshape(batch, routes, -1)
        route_present = event_mask.any(dim=-1)
        logits = self.route_gate(route_repr).squeeze(-1)
        logits = logits.masked_fill(~route_present, -1.0e9)
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(route_present, weights, torch.zeros_like(weights))
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = weights / denom
        return torch.sum(weights.unsqueeze(-1) * route_repr, dim=1)


class RelationalCNNEncoder(nn.Module):
    """Encode ``[batch, route, slot, channel]`` dense relational signals."""

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("layout tensor must have shape [batch, routes, slots, channels]")
        x = x.permute(0, 3, 1, 2)
        return self.norm(self.net(x).flatten(1))


BASIS_MODES = {
    "sum",
    "mamba",
    "cnn",
    "route_set",
    "route_attention",
    "deepset",
    "set_transformer",
    "perceiver",
    "relattn",
    "relational_attention",
    "masked_basis_attention",
    "bimamba",
    "multiscan_mamba",
    "route_mamba",
    "mixer",
    "ft_transformer",
    "basis_graph",
    "tcn",
}


class BasisEncoder(nn.Module):
    """Common interface for basis-token extractors."""

    def forward(self, tokens: Tensor, token_mask: Tensor) -> Tensor:  # pragma: no cover - protocol method
        raise NotImplementedError


class BasisMetadataMixin:
    """Register structural metadata for ``b=(route,column,aggregation,window)`` tokens."""

    def _init_basis_metadata(self, bases: Sequence[RelationalBasis]) -> None:
        basis_tuple = tuple(bases)
        self.num_bases = len(basis_tuple)
        self.route_names, route_ids = _ids_for_values(basis.route.name for basis in basis_tuple)
        self.table_names, table_ids = _ids_for_values(basis.end_table for basis in basis_tuple)
        self.column_names, column_ids = _ids_for_values((basis.column_name or "__row__") for basis in basis_tuple)
        self.aggregators, agg_ids = _ids_for_values(basis.aggregator for basis in basis_tuple)
        self.windows, window_ids = _ids_for_values(_basis_window_key(basis) for basis in basis_tuple)
        self.register_buffer("route_ids", torch.tensor(route_ids, dtype=torch.long))
        self.register_buffer("table_ids", torch.tensor(table_ids, dtype=torch.long))
        self.register_buffer("column_ids", torch.tensor(column_ids, dtype=torch.long))
        self.register_buffer("agg_ids", torch.tensor(agg_ids, dtype=torch.long))
        self.register_buffer("window_ids", torch.tensor(window_ids, dtype=torch.long))
        self.register_buffer(
            "hop_counts",
            torch.tensor([basis.route.hop_count for basis in basis_tuple], dtype=torch.long),
        )
        self.register_buffer("route_major_order", _order_tensor(basis_tuple, _route_major_sort_key))
        self.register_buffer("column_major_order", _order_tensor(basis_tuple, _column_major_sort_key))
        self.register_buffer("window_major_order", _order_tensor(basis_tuple, _window_major_sort_key))
        self.register_buffer("agg_major_order", _order_tensor(basis_tuple, _agg_major_sort_key))

    @property
    def num_routes(self) -> int:
        return len(self.route_names)


class AttentionPool(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        logits = self.score(x).squeeze(-1)
        logits = logits.masked_fill(~mask, -1.0e9)
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        return torch.sum(weights.unsqueeze(-1) * x, dim=1)


class SumBasisEncoder(BasisEncoder):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor, token_mask: Tensor) -> Tensor:
        mask_f = token_mask.to(dtype=tokens.dtype).unsqueeze(-1)
        return self.norm(torch.sum(tokens * mask_f, dim=1))


class BasisMambaEncoder(BasisEncoder):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        dropout: float,
        use_mamba_ssm: bool,
        allow_masked_mamba: bool,
    ) -> None:
        super().__init__()
        self.encoder = StackedMambaEncoder(
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
            use_mamba_ssm=use_mamba_ssm,
            allow_masked_mamba=allow_masked_mamba,
        )

    def forward(self, tokens: Tensor, token_mask: Tensor) -> Tensor:
        return masked_mean(self.encoder(tokens, token_mask), token_mask)


class CNNBasisEncoder(BasisEncoder):
    def __init__(self, bases: Sequence[RelationalBasis], d_model: int, dropout: float) -> None:
        super().__init__()
        self.layout = BasisLayout.from_bases(tuple(bases))
        self.encoder = RelationalCNNEncoder(d_model=d_model, dropout=dropout)

    def forward(self, tokens: Tensor, token_mask: Tensor) -> Tensor:
        return self.encoder(self.layout.tokens_to_tensor(tokens))


class RouteSetEncoder(BasisEncoder, BasisMetadataMixin):
    """Hierarchical route-set attention: pool basis tokens within routes, then routes."""

    def __init__(self, bases: Sequence[RelationalBasis], d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self._init_basis_metadata(bases)
        self.token_ffn = _ffn(d_model, dropout)
        self.within_route_pool = AttentionPool(d_model)
        self.route_pool = AttentionPool(d_model)

    def forward(self, tokens: Tensor, token_mask: Tensor) -> Tensor:
        tokens = tokens + self.token_ffn(tokens)
        route_reprs: list[Tensor] = []
        route_present: list[Tensor] = []
        route_ids = self.route_ids.to(device=tokens.device)
        for route_id in range(self.num_routes):
            mask = (route_ids == route_id).unsqueeze(0).expand(tokens.shape[0], -1) & token_mask
            route_reprs.append(self.within_route_pool(tokens, mask))
            route_present.append(mask.any(dim=1))
        routes = torch.stack(route_reprs, dim=1)
        present = torch.stack(route_present, dim=1)
        return self.route_pool(routes, present)


class DeepSetBasisEncoder(BasisEncoder):
    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.phi = _ffn(d_model, dropout)
        self.rho = _ffn(d_model, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor, token_mask: Tensor) -> Tensor:
        hidden = self.norm(tokens + self.phi(tokens))
        return self.rho(masked_mean(hidden, token_mask))


class SetTransformerBasisEncoder(BasisEncoder):
    def __init__(self, d_model: int, num_layers: int, dropout: float, num_heads: int = 4) -> None:
        super().__init__()
        heads = _valid_num_heads(d_model, num_heads)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=heads,
                    dim_feedforward=4 * d_model,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.pool = AttentionPool(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor, token_mask: Tensor) -> Tensor:
        x = tokens
        key_padding_mask = ~token_mask
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)
            x = torch.where(token_mask.unsqueeze(-1), x, torch.zeros_like(x))
        return self.norm(self.pool(x, token_mask))


class PerceiverBasisEncoder(BasisEncoder):
    def __init__(
        self,
        d_model: int,
        num_latents: int = 32,
        num_layers: int = 2,
        dropout: float = 0.0,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        heads = _valid_num_heads(d_model, num_heads)
        self.latents = nn.Parameter(torch.randn(num_latents, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=heads,
                    dim_feedforward=4 * d_model,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor, token_mask: Tensor) -> Tensor:
        batch = tokens.shape[0]
        latents = self.latents.unsqueeze(0).expand(batch, -1, -1)
        attn, _ = self.cross_attn(
            query=latents,
            key=tokens,
            value=tokens,
            key_padding_mask=~token_mask,
            need_weights=False,
        )
        latents = self.cross_norm(latents + attn)
        for layer in self.layers:
            latents = layer(latents)
        return self.norm(latents.mean(dim=1))


class RelationalBasisAttentionEncoder(BasisEncoder, BasisMetadataMixin):
    """Attention heads constrained by route/table/column/aggregation/window relations."""

    def __init__(
        self,
        bases: Sequence[RelationalBasis],
        d_model: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self._init_basis_metadata(bases)
        heads = _valid_num_heads(d_model, num_heads)
        relation_masks = self._relation_masks()
        self.register_buffer("relation_masks", relation_masks)
        self.attn_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True) for _ in range(relation_masks.shape[0])]
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.ffns = nn.ModuleList([_ffn(d_model, dropout) for _ in range(num_layers)])
        self.pool = AttentionPool(d_model)

    def forward(self, tokens: Tensor, token_mask: Tensor) -> Tensor:
        x = tokens
        relation_masks = self.relation_masks.to(device=tokens.device)
        key_padding_mask = ~token_mask
        for layer_index, attn_group in enumerate(self.attn_layers):
            outputs = []
            for relation_index, attn in enumerate(attn_group):
                out, _ = attn(
                    x,
                    x,
                    x,
                    attn_mask=relation_masks[relation_index],
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
                outputs.append(out)
            mixed = torch.stack(outputs, dim=0).mean(dim=0)
            x = self.norms[layer_index](x + mixed)
            x = x + self.ffns[layer_index](x)
            x = torch.where(token_mask.unsqueeze(-1), x, torch.zeros_like(x))
        return self.pool(x, token_mask)

    def _relation_masks(self) -> Tensor:
        ids = [self.route_ids, self.table_ids, self.column_ids, self.agg_ids, self.window_ids]
        masks = [torch.zeros(self.num_bases, self.num_bases, dtype=torch.bool)]
        for values in ids:
            same = values.view(-1, 1) == values.view(1, -1)
            masks.append(~same)
        return torch.stack(masks, dim=0)


class BiMambaBasisEncoder(BasisEncoder):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        dropout: float,
        use_mamba_ssm: bool,
        allow_masked_mamba: bool,
    ) -> None:
        super().__init__()
        self.forward_encoder = StackedMambaEncoder(d_model, num_layers, dropout, use_mamba_ssm, allow_masked_mamba)
        self.backward_encoder = StackedMambaEncoder(d_model, num_layers, dropout, use_mamba_ssm, allow_masked_mamba)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor, token_mask: Tensor) -> Tensor:
        forward = masked_mean(self.forward_encoder(tokens, token_mask), token_mask)
        rev_tokens = torch.flip(tokens, dims=(1,))
        rev_mask = torch.flip(token_mask, dims=(1,))
        backward = masked_mean(self.backward_encoder(rev_tokens, rev_mask), rev_mask)
        return self.norm(0.5 * (forward + backward))


class MultiScanMambaBasisEncoder(BasisEncoder, BasisMetadataMixin):
    def __init__(
        self,
        bases: Sequence[RelationalBasis],
        d_model: int,
        num_layers: int,
        dropout: float,
        use_mamba_ssm: bool,
        allow_masked_mamba: bool,
    ) -> None:
        super().__init__()
        self._init_basis_metadata(bases)
        self.register_buffer(
            "scan_orders",
            torch.stack([self.route_major_order, self.column_major_order, self.window_major_order, self.agg_major_order], dim=0),
        )
        self.encoders = nn.ModuleList(
            [
                StackedMambaEncoder(d_model, num_layers, dropout, use_mamba_ssm, allow_masked_mamba)
                for _ in range(self.scan_orders.shape[0])
            ]
        )
        self.scan_gate = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor, token_mask: Tensor) -> Tensor:
        reps = []
        orders = self.scan_orders.to(device=tokens.device)
        for order, encoder in zip(orders, self.encoders, strict=True):
            ordered_tokens = tokens.index_select(1, order)
            ordered_mask = token_mask.index_select(1, order)
            reps.append(masked_mean(encoder(ordered_tokens, ordered_mask), ordered_mask))
        stacked = torch.stack(reps, dim=1)
        weights = torch.softmax(self.scan_gate(stacked).squeeze(-1), dim=-1)
        return self.norm(torch.sum(weights.unsqueeze(-1) * stacked, dim=1))


class RouteMambaBasisEncoder(BasisEncoder, BasisMetadataMixin):
    def __init__(
        self,
        bases: Sequence[RelationalBasis],
        d_model: int,
        num_layers: int,
        dropout: float,
        use_mamba_ssm: bool,
        allow_masked_mamba: bool,
    ) -> None:
        super().__init__()
        self._init_basis_metadata(bases)
        self.route_encoder = StackedMambaEncoder(d_model, num_layers, dropout, use_mamba_ssm, allow_masked_mamba)
        self.route_pool = AttentionPool(d_model)

    def forward(self, tokens: Tensor, token_mask: Tensor) -> Tensor:
        route_ids = self.route_ids.to(device=tokens.device)
        route_reprs = []
        route_present = []
        for route_id in range(self.num_routes):
            indices = torch.nonzero(route_ids == route_id, as_tuple=False).flatten()
            route_tokens = tokens.index_select(1, indices)
            route_mask = token_mask.index_select(1, indices)
            route_reprs.append(masked_mean(self.route_encoder(route_tokens, route_mask), route_mask))
            route_present.append(route_mask.any(dim=1))
        routes = torch.stack(route_reprs, dim=1)
        present = torch.stack(route_present, dim=1)
        return self.route_pool(routes, present)


class BasisMixerEncoder(BasisEncoder):
    def __init__(self, num_bases: int, d_model: int, num_layers: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList([BasisMixerBlock(num_bases, d_model, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor, token_mask: Tensor) -> Tensor:
        x = torch.where(token_mask.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        for layer in self.layers:
            x = layer(x)
            x = torch.where(token_mask.unsqueeze(-1), x, torch.zeros_like(x))
        return self.norm(masked_mean(x, token_mask))


class BasisMixerBlock(nn.Module):
    def __init__(self, num_bases: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(num_bases)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_bases, num_bases),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_bases, num_bases),
        )
        self.channel_norm = nn.LayerNorm(d_model)
        self.channel_mlp = _ffn(d_model, dropout)

    def forward(self, x: Tensor) -> Tensor:
        token_view = x.transpose(1, 2)
        token_view = token_view + self.token_mlp(self.token_norm(token_view))
        x = token_view.transpose(1, 2)
        return x + self.channel_mlp(self.channel_norm(x))


class FTTransformerBasisEncoder(BasisEncoder):
    def __init__(self, d_model: int, num_layers: int, dropout: float, num_heads: int = 4) -> None:
        super().__init__()
        heads = _valid_num_heads(d_model, num_heads)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=heads,
                    dim_feedforward=4 * d_model,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor, token_mask: Tensor) -> Tensor:
        cls = self.cls.expand(tokens.shape[0], -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        cls_mask = torch.ones(tokens.shape[0], 1, dtype=torch.bool, device=tokens.device)
        mask = torch.cat([cls_mask, token_mask], dim=1)
        key_padding_mask = ~mask
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)
        return self.norm(x[:, 0])


class BasisTCNEncoder(BasisEncoder):
    def __init__(self, d_model: int, num_layers: int = 2, dropout: float = 0.0, kernel_size: int = 3) -> None:
        super().__init__()
        layers = []
        for layer_index in range(num_layers):
            dilation = 2**layer_index
            padding = dilation * (kernel_size - 1)
            layers.extend(
                [
                    nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding, dilation=dilation),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor, token_mask: Tensor) -> Tensor:
        x = tokens * token_mask.to(dtype=tokens.dtype).unsqueeze(-1)
        y = self.net(x.transpose(1, 2))
        y = y[:, :, : tokens.shape[1]].transpose(1, 2)
        return self.norm(masked_mean(y, token_mask))


class BasisGraphEncoder(BasisEncoder, BasisMetadataMixin):
    """Message passing over a small basis-token graph, not the row-level RDB graph."""

    def __init__(self, bases: Sequence[RelationalBasis], d_model: int, num_layers: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self._init_basis_metadata(bases)
        adjacency = self._adjacency()
        self.register_buffer("adjacency", adjacency)
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.ffns = nn.ModuleList([_ffn(d_model, dropout) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.pool = AttentionPool(d_model)

    def forward(self, tokens: Tensor, token_mask: Tensor) -> Tensor:
        adj = self.adjacency.to(device=tokens.device, dtype=tokens.dtype)
        x = tokens
        for linear, ffn, norm in zip(self.layers, self.ffns, self.norms, strict=True):
            message = torch.einsum("ij,bjd->bid", adj, linear(x))
            x = norm(x + message)
            x = x + ffn(x)
            x = torch.where(token_mask.unsqueeze(-1), x, torch.zeros_like(x))
        return self.pool(x, token_mask)

    def _adjacency(self) -> Tensor:
        relations = [
            self.route_ids,
            self.table_ids,
            self.column_ids,
            self.agg_ids,
            self.window_ids,
        ]
        adjacency = torch.eye(self.num_bases, dtype=torch.float32)
        for values in relations:
            adjacency = torch.maximum(adjacency, (values.view(-1, 1) == values.view(1, -1)).float())
        return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)


def build_basis_encoder(
    mode: str,
    bases: Sequence[RelationalBasis],
    d_model: int,
    num_layers: int,
    dropout: float,
    use_mamba_ssm: bool,
    allow_masked_mamba: bool,
) -> BasisEncoder:
    canonical = _canonical_basis_mode(mode)
    if canonical == "sum":
        return SumBasisEncoder(d_model)
    if canonical == "mamba":
        return BasisMambaEncoder(d_model, num_layers, dropout, use_mamba_ssm, allow_masked_mamba)
    if canonical == "cnn":
        return CNNBasisEncoder(bases, d_model, dropout)
    if canonical == "route_set":
        return RouteSetEncoder(bases, d_model, dropout)
    if canonical == "deepset":
        return DeepSetBasisEncoder(d_model, dropout)
    if canonical == "set_transformer":
        return SetTransformerBasisEncoder(d_model, num_layers, dropout)
    if canonical == "perceiver":
        return PerceiverBasisEncoder(d_model, num_layers=num_layers, dropout=dropout)
    if canonical == "relattn":
        return RelationalBasisAttentionEncoder(bases, d_model, num_layers, dropout)
    if canonical == "bimamba":
        return BiMambaBasisEncoder(d_model, num_layers, dropout, use_mamba_ssm, allow_masked_mamba)
    if canonical == "multiscan_mamba":
        return MultiScanMambaBasisEncoder(bases, d_model, num_layers, dropout, use_mamba_ssm, allow_masked_mamba)
    if canonical == "route_mamba":
        return RouteMambaBasisEncoder(bases, d_model, num_layers, dropout, use_mamba_ssm, allow_masked_mamba)
    if canonical == "mixer":
        return BasisMixerEncoder(len(bases), d_model, num_layers, dropout)
    if canonical == "ft_transformer":
        return FTTransformerBasisEncoder(d_model, num_layers, dropout)
    if canonical == "basis_graph":
        return BasisGraphEncoder(bases, d_model, num_layers, dropout)
    if canonical == "tcn":
        return BasisTCNEncoder(d_model, num_layers, dropout)
    raise ValueError(f"unknown basis_mode {mode!r}")


def _canonical_basis_mode(mode: str) -> str:
    aliases = {
        "route_attention": "route_set",
        "relational_attention": "relattn",
        "masked_basis_attention": "relattn",
    }
    return aliases.get(mode, mode)


def _ffn(d_model: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, 4 * d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(4 * d_model, d_model),
        nn.Dropout(dropout),
    )


def _valid_num_heads(d_model: int, preferred: int) -> int:
    for heads in range(min(preferred, d_model), 0, -1):
        if d_model % heads == 0:
            return heads
    return 1


def _ids_for_values(values: Iterable[str]) -> tuple[tuple[str, ...], list[int]]:
    names: list[str] = []
    lookup: dict[str, int] = {}
    ids: list[int] = []
    for value in values:
        if value not in lookup:
            lookup[value] = len(names)
            names.append(value)
        ids.append(lookup[value])
    return tuple(names), ids


def _basis_window_key(basis: RelationalBasis) -> str:
    if basis.window is None:
        return "all"
    return f"{int(basis.window.total_seconds() // 86_400)}d"


def _order_tensor(bases: Sequence[RelationalBasis], key_fn) -> Tensor:
    return torch.tensor(sorted(range(len(bases)), key=lambda index: key_fn(bases[index])), dtype=torch.long)


def _route_major_sort_key(basis: RelationalBasis) -> tuple:
    return (
        basis.route.hop_count,
        basis.route.table_path,
        basis.route.roles,
        basis.route.name,
        basis.column_name or "__row__",
        basis.aggregator,
        _basis_window_key(basis),
    )


def _column_major_sort_key(basis: RelationalBasis) -> tuple:
    return (
        basis.column_name or "__row__",
        basis.route.hop_count,
        basis.route.table_path,
        basis.aggregator,
        _basis_window_key(basis),
    )


def _window_major_sort_key(basis: RelationalBasis) -> tuple:
    return (
        _basis_window_key(basis),
        basis.route.hop_count,
        basis.route.table_path,
        basis.column_name or "__row__",
        basis.aggregator,
    )


def _agg_major_sort_key(basis: RelationalBasis) -> tuple:
    return (
        basis.aggregator,
        basis.route.hop_count,
        basis.route.table_path,
        basis.column_name or "__row__",
        _basis_window_key(basis),
    )


class RiftMambaModel(nn.Module):
    """End-to-end RIFT-Mamba predictor."""

    def __init__(
        self,
        bases: Iterable[RelationalBasis],
        d_model: int,
        output_dim: int,
        basis_layers: int = 2,
        event_dim: int | None = None,
        num_temporal_routes: int = 0,
        sequence_layers: int = 2,
        dropout: float = 0.0,
        basis_mode: str = "route_set",
        semantic_encoder: SchemaSemanticEncoder | None = None,
        semantic_matrix: np.ndarray | Tensor | None = None,
        use_mamba_ssm: bool = True,
        allow_masked_mamba: bool = True,
    ) -> None:
        super().__init__()
        if basis_mode not in BASIS_MODES:
            raise ValueError(f"basis_mode must be one of {sorted(BASIS_MODES)}")
        self.basis_mode = _canonical_basis_mode(basis_mode)
        basis_tuple = tuple(bases)
        self.basis_synthesizer = RelationalBasisSynthesizer(
            bases=basis_tuple,
            d_model=d_model,
            semantic_encoder=semantic_encoder,
            semantic_matrix=semantic_matrix,
        )
        self.basis_encoder = build_basis_encoder(
            mode=self.basis_mode,
            bases=basis_tuple,
            d_model=d_model,
            num_layers=basis_layers,
            dropout=dropout,
            use_mamba_ssm=use_mamba_ssm,
            allow_masked_mamba=allow_masked_mamba,
        )
        self.sequence_encoder: RouteWiseSequenceEncoder | None = None
        fusion_dim = d_model
        if event_dim is not None and num_temporal_routes > 0:
            self.sequence_encoder = RouteWiseSequenceEncoder(
                num_routes=num_temporal_routes,
                event_dim=event_dim,
                d_model=d_model,
                num_layers=sequence_layers,
                dropout=dropout,
                use_mamba_ssm=use_mamba_ssm,
                allow_masked_mamba=allow_masked_mamba,
            )
            fusion_dim += d_model
        self.feature_dim = fusion_dim
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

    def forward(
        self,
        alpha: Tensor,
        alpha_mask: Tensor,
        events: Tensor | None = None,
        event_mask: Tensor | None = None,
        return_features: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        features = self.encode(alpha, alpha_mask, events=events, event_mask=event_mask)
        logits = self.head(features)
        if return_features:
            return logits, features
        return logits

    def encode(
        self,
        alpha: Tensor,
        alpha_mask: Tensor,
        events: Tensor | None = None,
        event_mask: Tensor | None = None,
    ) -> Tensor:
        """Return fused RIFT-Mamba features before the prediction head."""

        basis_tokens = self.basis_synthesizer(alpha, alpha_mask)
        basis_padding_mask = torch.ones(alpha.shape, dtype=torch.bool, device=alpha.device)
        h_basis = self.basis_encoder(basis_tokens, basis_padding_mask)

        pieces = [h_basis]
        if self.sequence_encoder is not None:
            if events is None or event_mask is None:
                raise ValueError("events and event_mask are required when sequence_encoder is enabled")
            pieces.append(self.sequence_encoder(events, event_mask))
        return torch.cat(pieces, dim=-1)


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    mask_f = mask.to(dtype=x.dtype).unsqueeze(-1)
    total = torch.sum(x * mask_f, dim=1)
    denom = mask_f.sum(dim=1).clamp_min(1e-6)
    return total / denom


def masked_last(x: Tensor, mask: Tensor) -> Tensor:
    positions = torch.arange(mask.shape[1], device=mask.device).view(1, -1)
    last_indices = torch.where(mask, positions, torch.full_like(positions, -1)).max(dim=1).values
    batch_indices = torch.arange(x.shape[0], device=x.device)
    gathered = x[batch_indices, torch.clamp(last_indices, min=0)]
    return torch.where(last_indices.unsqueeze(-1) >= 0, gathered, torch.zeros_like(gathered))
