from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


TransitionStep = Callable[[int, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True, slots=True)
class ViterbiResult:
    path_indices: tuple[int, ...]
    cost: float
    reachable_counts: tuple[int, ...]

    @property
    def success(self) -> bool:
        return bool(self.path_indices)


def solve_viterbi(
    initial_costs: np.ndarray,
    num_layers: int,
    transition_step: TransitionStep,
) -> ViterbiResult:
    """Find the minimum-cost path through an equal-width layered graph."""
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")

    previous_costs = np.asarray(initial_costs, dtype=float)
    if previous_costs.ndim != 1 or previous_costs.size == 0:
        raise ValueError("initial_costs must be a non-empty vector")

    state_count = previous_costs.size
    reachable_counts = [int(np.count_nonzero(np.isfinite(previous_costs)))]
    if reachable_counts[0] == 0:
        return ViterbiResult((), float("inf"), tuple(reachable_counts))

    predecessors: list[np.ndarray] = []
    for layer_index in range(1, num_layers):
        next_costs, previous_indices = transition_step(layer_index, previous_costs)
        next_costs = np.asarray(next_costs, dtype=float)
        previous_indices = np.asarray(previous_indices, dtype=np.int64)
        if next_costs.shape != (state_count,) or previous_indices.shape != (state_count,):
            raise ValueError(
                f"layer {layer_index} must return two ({state_count},) vectors"
            )

        reachable = np.isfinite(next_costs)
        invalid = reachable & (
            (previous_indices < 0) | (previous_indices >= state_count)
        )
        if np.any(invalid):
            raise ValueError(
                f"layer {layer_index} contains a reachable state with an invalid predecessor"
            )

        previous_indices = previous_indices.copy()
        previous_indices[~reachable] = -1
        predecessors.append(previous_indices)
        previous_costs = next_costs
        reachable_counts.append(int(np.count_nonzero(reachable)))
        if reachable_counts[-1] == 0:
            return ViterbiResult((), float("inf"), tuple(reachable_counts))

    current = int(np.argmin(previous_costs))
    path = [current]
    for previous_indices in reversed(predecessors):
        current = int(previous_indices[current])
        path.append(current)
    path.reverse()
    return ViterbiResult(
        tuple(path),
        float(np.min(previous_costs)),
        tuple(reachable_counts),
    )
