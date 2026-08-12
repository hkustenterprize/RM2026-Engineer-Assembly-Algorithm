from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class JointTrajectory:
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    timestamps: np.ndarray

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=float)
        velocities = np.asarray(self.velocities, dtype=float)
        accelerations = np.asarray(self.accelerations, dtype=float)
        timestamps = np.asarray(self.timestamps, dtype=float)
        if positions.ndim != 2:
            raise ValueError(f"positions must have shape (B, dof), got {positions.shape}")
        if velocities.shape != positions.shape or accelerations.shape != positions.shape:
            raise ValueError("trajectory positions, velocities, and accelerations must have identical shapes")
        if timestamps.shape != (len(positions),) or np.any(np.diff(timestamps) <= 0.0):
            raise ValueError("timestamps must be strictly increasing with one value per trajectory point")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "velocities", velocities)
        object.__setattr__(self, "accelerations", accelerations)
        object.__setattr__(self, "timestamps", timestamps)


class FixedDurationParameterizer:
    def __init__(self, duration_s: float | None = None, **_: object) -> None:
        self.duration_s = None if duration_s is None else float(duration_s)

    def parameterize(
        self,
        waypoints: np.ndarray,
        *,
        sample_dt: float,
        duration_s: float | None = None,
        **_: object,
    ) -> JointTrajectory:
        duration = self.duration_s if duration_s is None else float(duration_s)
        if duration is None or duration <= 0.0:
            raise ValueError("duration_s must be positive")
        points = np.asarray(waypoints, dtype=float)
        if points.ndim != 2 or len(points) < 2:
            raise ValueError(f"waypoints must have shape (B, dof) with B >= 2, got {points.shape}")
        return sample_quintic_trajectory(
            points,
            np.linspace(0.0, duration, len(points)),
            sample_dt,
        )


def sample_quintic_trajectory(
    waypoints: np.ndarray,
    waypoint_times: np.ndarray,
    sample_dt: float,
) -> JointTrajectory:
    positions = np.asarray(waypoints, dtype=float)
    times = np.asarray(waypoint_times, dtype=float)
    dt = float(sample_dt)
    if positions.ndim != 2 or len(positions) < 2:
        raise ValueError(f"waypoints must have shape (B, dof) with B >= 2, got {positions.shape}")
    if times.shape != (len(positions),) or np.any(np.diff(times) <= 0.0):
        raise ValueError("waypoint_times must be strictly increasing with one value per waypoint")
    if dt <= 0.0:
        raise ValueError("sample_dt must be positive")

    waypoint_velocities, waypoint_accelerations = _estimate_derivatives(positions, times)
    sample_times = _sample_times(times, dt)
    segments = np.clip(np.searchsorted(times, sample_times, side="right") - 1, 0, len(positions) - 2)
    start_times = times[segments]
    durations = times[segments + 1] - start_times
    progress = (sample_times - start_times) / durations
    q0 = positions[segments]
    q1 = positions[segments + 1]
    v0 = waypoint_velocities[segments]
    v1 = waypoint_velocities[segments + 1]
    a0 = waypoint_accelerations[segments]
    a1 = waypoint_accelerations[segments + 1]
    h = durations[:, None]
    c0 = q0
    c1 = h * v0
    c2 = 0.5 * h**2 * a0
    displacement = q1 - (c0 + c1 + c2)
    velocity_delta = h * v1 - (c1 + 2.0 * c2)
    acceleration_delta = h**2 * a1 - 2.0 * c2
    c5 = 0.5 * (acceleration_delta + 12.0 * displacement - 6.0 * velocity_delta)
    c4 = velocity_delta - 3.0 * displacement - 2.0 * c5
    c3 = displacement - c4 - c5
    u = progress[:, None]
    sampled_positions = c0 + c1 * u + c2 * u**2 + c3 * u**3 + c4 * u**4 + c5 * u**5
    sampled_velocities = (
        c1 + 2.0 * c2 * u + 3.0 * c3 * u**2 + 4.0 * c4 * u**3 + 5.0 * c5 * u**4
    ) / h
    sampled_accelerations = (
        2.0 * c2 + 6.0 * c3 * u + 12.0 * c4 * u**2 + 20.0 * c5 * u**3
    ) / h**2
    sampled_positions[-1] = positions[-1]
    sampled_velocities[-1] = waypoint_velocities[-1]
    sampled_accelerations[-1] = waypoint_accelerations[-1]
    return JointTrajectory(sampled_positions, sampled_velocities, sampled_accelerations, sample_times)


def _estimate_derivatives(positions: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    velocities = np.zeros_like(positions)
    accelerations = np.zeros_like(positions)
    if len(positions) > 2:
        velocities[1:-1] = (positions[2:] - positions[:-2]) / (
            timestamps[2:] - timestamps[:-2]
        )[:, None]
        accelerations[1:-1] = 2.0 * (
            (positions[2:] - positions[1:-1]) / (timestamps[2:] - timestamps[1:-1])[:, None]
            - (positions[1:-1] - positions[:-2]) / (timestamps[1:-1] - timestamps[:-2])[:, None]
        ) / (timestamps[2:] - timestamps[:-2])[:, None]
    return velocities, accelerations


def _sample_times(waypoint_times: np.ndarray, sample_dt: float) -> np.ndarray:
    start = float(waypoint_times[0])
    end = float(waypoint_times[-1])
    samples = np.arange(start, end, sample_dt, dtype=float)
    if samples.size == 0 or samples[0] != start:
        samples = np.concatenate(([start], samples))
    if samples[-1] != end:
        samples = np.concatenate((samples, [end]))
    return samples
