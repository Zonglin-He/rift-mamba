"""Baseline interfaces for RDL experiments.

Built-in baselines cover DFS-style neural models. Methods such as GraphSAGE,
RelGNN, RelGT and RT are exposed as external command wrappers because their
faithful implementations depend on separate official codebases.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import subprocess
from typing import Mapping, Protocol

import torch
from torch import Tensor, nn


class Baseline(Protocol):
    name: str

    def fit(self, *args, **kwargs):
        ...

    def predict(self, *args, **kwargs):
        ...


class DFSMLPBaseline(nn.Module):
    """A direct neural baseline over extracted alpha(q) DFS coefficients."""

    name = "dfs_mlp"

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, alpha: Tensor, mask: Tensor | None = None) -> Tensor:
        if mask is not None:
            alpha = alpha * mask.to(dtype=alpha.dtype)
        return self.net(alpha)


@dataclass(frozen=True)
class ExternalBaseline:
    """Run a third-party baseline through a configured command."""

    name: str
    command: tuple[str, ...]
    workdir: str | None = None

    def run(self, config: Mapping[str, object], output_path: str | Path) -> dict:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        config_path = output.with_suffix(".config.json")
        config_path.write_text(json.dumps(dict(config), indent=2), encoding="utf-8")
        completed = subprocess.run(
            [*self.command, "--config", str(config_path), "--output", str(output)],
            cwd=self.workdir,
            check=True,
            capture_output=True,
            text=True,
        )
        if output.exists():
            return json.loads(output.read_text(encoding="utf-8"))
        return {"stdout": completed.stdout, "stderr": completed.stderr}


def default_baseline_registry() -> dict[str, str]:
    """Names used for experiment tables and external adapters."""

    return {
        "dfs_mlp": "Built-in DFS coefficient MLP baseline.",
        "dfs_lightgbm": "External LightGBM/DFS baseline adapter.",
        "graphsage_rdl": "External relational GraphSAGE baseline adapter.",
        "relgnn": "External RelGNN atomic-route baseline adapter.",
        "relgt": "External Relational Graph Transformer baseline adapter.",
        "rt": "External Relational Transformer baseline adapter.",
        "griffin": "External Griffin baseline adapter.",
    }
