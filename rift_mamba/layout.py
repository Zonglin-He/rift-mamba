"""Route x slot dense layout for relational basis signals."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rift_mamba.basis import RelationalBasis


@dataclass(frozen=True)
class BasisLayout:
    """Map basis tokens into a stable ``[route, slot, channel]`` tensor."""

    bases: tuple[RelationalBasis, ...]
    route_names: tuple[str, ...]
    slots: tuple[tuple[str, str, str], ...]
    basis_to_position: tuple[tuple[int, int], ...]

    @classmethod
    def from_bases(cls, bases: tuple[RelationalBasis, ...]) -> "BasisLayout":
        route_names = _ordered_unique(basis.route.name for basis in bases)
        slots = _ordered_unique(_slot_key(basis) for basis in bases)
        route_index = {name: index for index, name in enumerate(route_names)}
        slot_index = {slot: index for index, slot in enumerate(slots)}
        positions = tuple((route_index[basis.route.name], slot_index[_slot_key(basis)]) for basis in bases)
        return cls(bases=bases, route_names=route_names, slots=slots, basis_to_position=positions)

    @property
    def num_routes(self) -> int:
        return len(self.route_names)

    @property
    def num_slots(self) -> int:
        return len(self.slots)

    def tokens_to_tensor(self, tokens: Tensor) -> Tensor:
        """Scatter basis tokens ``[batch, basis, channel]`` to ``[batch, route, slot, channel]``."""

        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [batch, num_bases, channels]")
        if tokens.shape[1] != len(self.bases):
            raise ValueError("token basis dimension does not match layout")
        output = tokens.new_zeros(tokens.shape[0], self.num_routes, self.num_slots, tokens.shape[2])
        for basis_index, (route_index, slot_index) in enumerate(self.basis_to_position):
            output[:, route_index, slot_index, :] += tokens[:, basis_index, :]
        return output


def _slot_key(basis: RelationalBasis) -> tuple[str, str, str]:
    window = "all" if basis.window is None else f"{int(basis.window.total_seconds() // 86400)}d"
    return (basis.column_name or "__row__", basis.aggregator, window)


def _ordered_unique(values) -> tuple:
    seen = set()
    output = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return tuple(output)
