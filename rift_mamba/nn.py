"""Neural modules for RIFT-Mamba."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from rift_mamba.basis import RelationalBasis


class RelationalBasisSynthesizer(nn.Module):
    """Compose alpha_b(q) with learnable basis and missing-value tensors.

    For each basis token:

        e_b(q) = m_b alpha_b(q) Psi_b + (1 - m_b) Omega_b

    The output is a dense sequence of basis-indexed neural tokens.
    """

    def __init__(self, bases: Iterable[RelationalBasis], d_model: int) -> None:
        super().__init__()
        self.bases = tuple(bases)
        self.num_bases = len(self.bases)
        self.d_model = d_model
        self.basis = nn.Parameter(torch.empty(self.num_bases, d_model))
        self.missing_basis = nn.Parameter(torch.empty(self.num_bases, d_model))
        self.alpha_scale = nn.Parameter(torch.ones(self.num_bases))
        self.alpha_bias = nn.Parameter(torch.zeros(self.num_bases))
        self.mask_embedding = nn.Embedding(2, d_model)
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

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fallback = CausalGatedSSMBlock(d_model=d_model, dropout=dropout)
        try:
            from mamba_ssm import Mamba  # type: ignore
        except Exception:
            self.impl: nn.Module = self.fallback
            self.uses_mamba_ssm = False
        else:
            self.impl = Mamba(d_model=d_model)
            self.dropout = nn.Dropout(dropout)
            self.uses_mamba_ssm = True

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        if mask is not None and not bool(mask.all()):
            return self.fallback(x, mask)
        if self.uses_mamba_ssm:
            y = self.impl(x)
            if mask is not None:
                y = y * mask.to(dtype=y.dtype).unsqueeze(-1)
            return self.dropout(y)  # type: ignore[attr-defined]
        return self.impl(x, mask)


class StackedMambaEncoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList([MambaBlock(d_model=d_model, dropout=dropout) for _ in range(num_layers)])
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
    ) -> None:
        super().__init__()
        self.num_routes = num_routes
        self.event_proj = nn.Linear(event_dim, d_model)
        self.route_embedding = nn.Embedding(num_routes, d_model)
        self.encoder = StackedMambaEncoder(d_model=d_model, num_layers=num_layers, dropout=dropout)
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
        basis_mode: str = "mamba",
    ) -> None:
        super().__init__()
        if basis_mode not in {"mamba", "sum"}:
            raise ValueError("basis_mode must be 'mamba' or 'sum'")
        self.basis_mode = basis_mode
        self.basis_synthesizer = RelationalBasisSynthesizer(bases=bases, d_model=d_model)
        self.basis_encoder = StackedMambaEncoder(d_model=d_model, num_layers=basis_layers, dropout=dropout)
        self.sum_norm = nn.LayerNorm(d_model)
        self.sequence_encoder: RouteWiseSequenceEncoder | None = None
        fusion_dim = d_model
        if event_dim is not None and num_temporal_routes > 0:
            self.sequence_encoder = RouteWiseSequenceEncoder(
                num_routes=num_temporal_routes,
                event_dim=event_dim,
                d_model=d_model,
                num_layers=sequence_layers,
                dropout=dropout,
            )
            fusion_dim += d_model
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
    ) -> Tensor:
        basis_tokens = self.basis_synthesizer(alpha, alpha_mask)
        if self.basis_mode == "sum":
            h_basis = self.sum_norm(basis_tokens.sum(dim=1))
        else:
            basis_padding_mask = torch.ones(
                alpha.shape,
                dtype=torch.bool,
                device=alpha.device,
            )
            basis_encoded = self.basis_encoder(basis_tokens, basis_padding_mask)
            h_basis = masked_mean(basis_encoded, basis_padding_mask)

        pieces = [h_basis]
        if self.sequence_encoder is not None:
            if events is None or event_mask is None:
                raise ValueError("events and event_mask are required when sequence_encoder is enabled")
            pieces.append(self.sequence_encoder(events, event_mask))
        return self.head(torch.cat(pieces, dim=-1))


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
