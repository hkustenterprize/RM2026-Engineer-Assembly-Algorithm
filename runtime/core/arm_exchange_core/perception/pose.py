from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np

from .detector import KeypointObservation


class PnPEstimator:
    """Estimate the camera-from-target transform from named image keypoints."""

    def __init__(
        self,
        *,
        object_points: Mapping[str, list[float] | np.ndarray],
        use_ransac: bool = False,
        ransac_reprojection_error_px: float = 8.0,
        ransac_confidence: float = 0.99,
        ransac_iterations: int = 100,
    ) -> None:
        self.object_points = {
            str(name): np.asarray(point, dtype=float)
            for name, point in object_points.items()
        }
        if not self.object_points or any(point.shape != (3,) for point in self.object_points.values()):
            raise ValueError("object_points must map keypoint names to three-dimensional points")
        self.use_ransac = bool(use_ransac)
        self.ransac_reprojection_error_px = float(ransac_reprojection_error_px)
        self.ransac_confidence = float(ransac_confidence)
        self.ransac_iterations = int(ransac_iterations)

    def estimate(
        self,
        observation: KeypointObservation,
        camera_matrix: np.ndarray,
        distortion: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        camera = np.asarray(camera_matrix, dtype=float)
        if camera.shape != (3, 3):
            raise ValueError(f"camera_matrix must have shape (3, 3), got {camera.shape}")
        dist = np.zeros(5, dtype=float) if distortion is None else np.asarray(distortion, dtype=float).reshape(-1)
        object_points, image_points = self._matched_points(observation)

        inliers = np.arange(len(object_points))
        if self.use_ransac:
            ok, _, _, selected = cv2.solvePnPRansac(
                object_points,
                image_points,
                camera,
                dist,
                iterationsCount=self.ransac_iterations,
                reprojectionError=self.ransac_reprojection_error_px,
                confidence=self.ransac_confidence,
                flags=cv2.SOLVEPNP_SQPNP,
            )
            if not ok or selected is None or len(selected) < 4:
                raise RuntimeError("RANSAC produced fewer than four PnP inliers")
            inliers = selected.reshape(-1)

        solve_object = object_points[inliers]
        solve_image = image_points[inliers]
        ok, rotations, translations, *_ = cv2.solvePnPGeneric(
            solve_object,
            solve_image,
            camera,
            dist,
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if not ok:
            raise RuntimeError("SQPnP failed")

        candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
        for rotation_vector, translation_vector in zip(rotations, translations, strict=True):
            rvec = np.asarray(rotation_vector, dtype=float).reshape(3, 1)
            tvec = np.asarray(translation_vector, dtype=float).reshape(3, 1)
            if np.min(self._camera_depths(solve_object, rvec, tvec)) <= 0.0:
                continue
            rvec, tvec = cv2.solvePnPRefineLM(
                solve_object,
                solve_image,
                camera,
                dist,
                rvec,
                tvec,
            )
            if np.min(self._camera_depths(solve_object, rvec, tvec)) <= 0.0:
                continue
            error = self._mean_reprojection_error(
                solve_object,
                solve_image,
                camera,
                dist,
                rvec,
                tvec,
            )
            candidates.append((error, rvec, tvec))

        if not candidates:
            raise RuntimeError("PnP rejected every candidate behind the camera")

        _, rvec, tvec = min(candidates, key=lambda candidate: candidate[0])
        reprojection_error = self._mean_reprojection_error(
            object_points,
            image_points,
            camera,
            dist,
            rvec,
            tvec,
        )
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = cv2.Rodrigues(rvec)[0]
        transform[:3, 3] = tvec.reshape(3)
        return transform, reprojection_error

    def _matched_points(self, observation: KeypointObservation) -> tuple[np.ndarray, np.ndarray]:
        matched = [
            (self.object_points[name], point)
            for name, point in zip(
                observation.keypoint_names,
                observation.keypoints_2d,
                strict=True,
            )
            if name in self.object_points
        ]
        if len(matched) < 4:
            raise ValueError(f"PnP requires at least four matched keypoints, got {len(matched)}")
        object_points, image_points = zip(*matched, strict=True)
        return np.asarray(object_points), np.asarray(image_points)

    @staticmethod
    def _camera_depths(
        object_points: np.ndarray,
        rotation_vector: np.ndarray,
        translation_vector: np.ndarray,
    ) -> np.ndarray:
        rotation = cv2.Rodrigues(rotation_vector)[0]
        camera_points = object_points @ rotation.T + translation_vector.reshape(1, 3)
        return camera_points[:, 2]

    @staticmethod
    def _mean_reprojection_error(
        object_points: np.ndarray,
        image_points: np.ndarray,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        rotation_vector: np.ndarray,
        translation_vector: np.ndarray,
    ) -> float:
        projected = cv2.projectPoints(
            object_points,
            rotation_vector,
            translation_vector,
            camera_matrix,
            distortion,
        )[0].reshape(-1, 2)
        return float(np.mean(np.linalg.norm(projected - image_points, axis=1)))
