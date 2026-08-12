from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _wrap_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True, slots=True)
class JointSpace:
    lower: np.ndarray
    upper: np.ndarray
    continuous: np.ndarray
    dof: int = field(init=False)

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        continuous = np.asarray(self.continuous, dtype=bool)
        if lower.ndim != 1 or upper.shape != lower.shape or continuous.shape != lower.shape:
            raise ValueError("joint limits and continuous mask must be one-dimensional and equally sized")
        if np.any(~continuous & (~np.isfinite(lower) | ~np.isfinite(upper))):
            raise ValueError("bounded joints require finite lower and upper limits")
        if np.any(~continuous & (lower > upper)):
            raise ValueError("joint lower limits must not exceed upper limits")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "continuous", continuous)
        object.__setattr__(self, "dof", len(lower))

    @classmethod
    def from_config(cls, config: dict) -> "JointSpace":
        lower_raw = tuple(config["lower"])
        upper_raw = tuple(config["upper"])
        if len(lower_raw) != len(upper_raw):
            raise ValueError("joint lower and upper limits must have equal length")
        continuous = np.asarray(
            [lower is None and upper is None for lower, upper in zip(lower_raw, upper_raw, strict=True)]
        )
        lower = np.asarray([-np.inf if value is None else float(value) for value in lower_raw])
        upper = np.asarray([np.inf if value is None else float(value) for value in upper_raw])
        return cls(lower=lower, upper=upper, continuous=continuous)

    @property
    def bounded(self) -> np.ndarray:
        return ~self.continuous

    def _validate(self, joints: np.ndarray, name: str) -> np.ndarray:
        values = np.asarray(joints, dtype=float)
        if values.ndim < 2 or values.shape[-1] != self.dof:
            raise ValueError(f"{name} must have shape (..., B, {self.dof}), got {values.shape}")
        return values

    def delta(self, target: np.ndarray, source: np.ndarray) -> np.ndarray:
        target_values = self._validate(target, "target")
        source_values = self._validate(source, "source")
        delta = np.asarray(target_values - source_values, dtype=float)
        delta[..., self.continuous] = _wrap_pi(delta[..., self.continuous])
        return delta

    def align_to_reference(self, joints: np.ndarray, reference: np.ndarray) -> np.ndarray:
        aligned = self._validate(joints, "joints").copy()
        reference_values = self._validate(reference, "reference")
        aligned[..., self.continuous] += 2.0 * np.pi * np.round(
            (reference_values[..., self.continuous] - aligned[..., self.continuous]) / (2.0 * np.pi)
        )
        return aligned

    def align_trajectory(self, trajectory: np.ndarray, reference: np.ndarray) -> np.ndarray:
        aligned = self._validate(trajectory, "trajectory").copy()
        if aligned.ndim != 2:
            raise ValueError(f"trajectory must have shape (B, {self.dof}), got {aligned.shape}")
        reference_values = self._validate(reference, "reference")
        if reference_values.shape != (1, self.dof):
            raise ValueError(f"reference must have shape (1, {self.dof}), got {reference_values.shape}")
        for index in np.flatnonzero(self.continuous):
            aligned[:, index] = np.unwrap(aligned[:, index])
            aligned[:, index] += 2.0 * np.pi * np.round(
                (reference_values[0, index] - aligned[0, index]) / (2.0 * np.pi)
            )
        return aligned

    def wrap(self, joints: np.ndarray) -> np.ndarray:
        wrapped = self._validate(joints, "joints").copy()
        wrapped[..., self.continuous] = _wrap_pi(wrapped[..., self.continuous])
        return wrapped

    def normalize(self, joints: np.ndarray, *, tolerance: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
        values = self._validate(joints, "joints")
        flat = values.reshape(-1, self.dof)
        normalized = flat.copy()
        valid = np.ones(len(flat), dtype=bool)
        period = 2.0 * np.pi
        for index in np.flatnonzero(self.bounded):
            current = normalized[:, index]
            lower = self.lower[index]
            upper = self.upper[index]
            minimum_turns = np.ceil((lower - current - tolerance) / period).astype(np.int64)
            maximum_turns = np.floor((upper - current + tolerance) / period).astype(np.int64)
            feasible = minimum_turns <= maximum_turns
            valid &= feasible
            center_turns = np.rint((0.5 * (lower + upper) - current) / period).astype(np.int64)
            turns = np.clip(center_turns, minimum_turns, maximum_turns)
            normalized[:, index] = np.where(feasible, current + period * turns, current)
        return normalized.reshape(values.shape), valid.reshape(values.shape[:-1])

    def limit_penalty(self, joints: np.ndarray, margin: float, weight: float) -> np.ndarray:
        values = self._validate(joints, "joints")
        if margin <= 0.0 or weight <= 0.0 or not np.any(self.bounded):
            return np.zeros(values.shape[:-1], dtype=float)
        distance = np.minimum(
            values[..., self.bounded] - self.lower[self.bounded],
            self.upper[self.bounded] - values[..., self.bounded],
        )
        violation = np.maximum(0.0, float(margin) - distance)
        return float(weight) * np.sum(violation * violation, axis=-1)
