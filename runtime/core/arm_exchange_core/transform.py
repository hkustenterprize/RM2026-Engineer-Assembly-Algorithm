from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def validate_transforms(transforms: np.ndarray, *, name: str = "transforms") -> np.ndarray:
    """Validate a batch of homogeneous transforms with shape ``(B, 4, 4)``."""
    values = np.asarray(transforms, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape (B, 4, 4), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    expected_bottom = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), (values.shape[0], 4))
    if not np.allclose(values[:, 3, :], expected_bottom, atol=1e-9):
        raise ValueError(f"{name} must contain homogeneous transforms")
    return values


def rotations_from_quaternions(quaternions: np.ndarray) -> np.ndarray:
    """Convert ``(B, 4)`` quaternions in ``[w, x, y, z]`` order to rotations."""
    values = np.asarray(quaternions, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"quaternions must have shape (B, 4), got {values.shape}")
    if np.any(np.linalg.norm(values, axis=1) <= 1e-12):
        raise ValueError("quaternions must be non-zero")
    return Rotation.from_quat(values[:, [1, 2, 3, 0]]).as_matrix()


def quaternions_from_rotations(rotations: np.ndarray) -> np.ndarray:
    """Convert ``(B, 3, 3)`` rotations to quaternions in ``[w, x, y, z]`` order."""
    matrices = np.asarray(rotations, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError(f"rotations must have shape (B, 3, 3), got {matrices.shape}")

    xyzw = Rotation.from_matrix(matrices).as_quat()
    return xyzw[:, [3, 0, 1, 2]]
