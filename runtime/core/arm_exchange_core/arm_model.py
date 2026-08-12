from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .joint_space import JointSpace
from .transform import validate_transforms


@dataclass(frozen=True, slots=True)
class ForwardKinematics:
    link_transforms: np.ndarray
    tcp_transforms: np.ndarray

    @property
    def link_origins(self) -> np.ndarray:
        return self.link_transforms[:, :, :3, 3]

    @property
    def link_rotations(self) -> np.ndarray:
        return self.link_transforms[:, :, :3, :3]


@dataclass(frozen=True, slots=True)
class ArmModel:
    a: np.ndarray
    alpha: np.ndarray
    d: np.ndarray
    theta_offset: np.ndarray
    tool_offset: np.ndarray
    joint_space: JointSpace
    masses: np.ndarray
    centers_of_mass: np.ndarray
    inertias: np.ndarray

    def __post_init__(self) -> None:
        for name in ("a", "alpha", "d", "theta_offset"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (6,):
                raise ValueError(f"{name} must have shape (6,), got {values.shape}")
            object.__setattr__(self, name, values)
        tool_offset = np.asarray(self.tool_offset, dtype=float)
        masses = np.asarray(self.masses, dtype=float)
        centers = np.asarray(self.centers_of_mass, dtype=float)
        inertias = np.asarray(self.inertias, dtype=float)
        if tool_offset.shape != (3,):
            raise ValueError(f"tool_offset must have shape (3,), got {tool_offset.shape}")
        if masses.shape != (6,) or centers.shape != (6, 3) or inertias.shape != (6, 3, 3):
            raise ValueError("dynamics parameters must have shapes (6,), (6, 3), and (6, 3, 3)")
        if self.joint_space.dof != 6:
            raise ValueError("ArmModel requires a six-dimensional JointSpace")
        object.__setattr__(self, "tool_offset", tool_offset)
        object.__setattr__(self, "masses", masses)
        object.__setattr__(self, "centers_of_mass", centers)
        object.__setattr__(self, "inertias", inertias)

    @classmethod
    def from_config(cls, config: dict) -> "ArmModel":
        dh = config["dh"]
        dynamics = config["dynamics"]
        if len(dynamics) != 6:
            raise ValueError("arm.dynamics must describe six links")
        return cls(
            a=np.asarray(dh["a"], dtype=float),
            alpha=np.asarray(dh["alpha"], dtype=float),
            d=np.asarray(dh["d"], dtype=float),
            theta_offset=np.asarray(dh["theta_offset"], dtype=float),
            tool_offset=np.asarray(config["tool"]["tcp_offset_6_tcp"], dtype=float),
            joint_space=JointSpace.from_config(config["joint_limits"]),
            masses=np.asarray([link["mass"] for link in dynamics], dtype=float),
            centers_of_mass=np.asarray([link["ipos"] for link in dynamics], dtype=float),
            inertias=np.asarray([link["inertia"] for link in dynamics], dtype=float),
        )

    def solve_ik(
        self,
        target_tcp: np.ndarray,
        branches: tuple[int, ...] | list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve analytic IK for ``(B, 4, 4)`` TCP targets."""
        targets = validate_transforms(target_tcp, name="target_tcp")
        branch_ids = self._normalize_branches(branches)
        rotations = targets[:, :3, :3]
        positions = targets[:, :3, 3] - np.einsum("bij,j->bi", rotations, self.tool_offset)
        batch_size = len(targets)
        joints = np.zeros((batch_size, len(branch_ids), 6), dtype=float)
        valid = np.zeros((batch_size, len(branch_ids)), dtype=bool)

        a2 = self.a[2]
        a3 = self.a[3]
        d4 = self.d[3]
        coefficient_a = 2.0 * a2 * a3
        coefficient_b = -2.0 * a2 * d4
        x, y, z = positions.T
        radius_squared = np.sum(positions * positions, axis=1)
        workspace_valid = (radius_squared >= 0.055**2) & (radius_squared <= 0.8464)
        coefficient_c = a2**2 + a3**2 + d4**2 - radius_squared
        discriminant = coefficient_b**2 - coefficient_c**2 + coefficient_a**2
        workspace_valid &= discriminant >= -1e-9
        theta1 = np.arctan2(y, x) - np.pi
        theta1_flipped = theta1 + np.pi
        branch_output = {branch: index for index, branch in enumerate(branch_ids)}
        signs = ((1, 1), (-1, 1), (1, -1), (-1, -1))

        for sign_index in sorted({branch // 2 for branch in branch_ids}):
            sign1, sign2 = signs[sign_index]
            theta3 = 2.0 * np.arctan2(
                -(coefficient_b + sign1 * np.sqrt(np.maximum(discriminant, 0.0))),
                -(coefficient_a - coefficient_c),
            )
            f1 = a3 * np.cos(theta3) - d4 * np.sin(theta3) + a2
            f2 = -a3 * np.sin(theta3) - d4 * np.cos(theta3)
            second_discriminant = f1**2 + f2**2 - z**2
            theta2 = 2.0 * np.arctan2(
                f1 + sign2 * np.sqrt(np.maximum(second_discriminant, 0.0)),
                f2 - z,
            )
            radial = np.cos(theta2) * f1 - np.sin(theta2) * f2
            tolerance = 1e-4
            first_valid = (
                (np.abs(np.cos(theta1) * radial - x) < tolerance)
                & (np.abs(np.sin(theta1) * radial - y) < tolerance)
            )
            flipped_valid = (
                (np.abs(np.cos(theta1_flipped) * radial - x) < tolerance)
                & (np.abs(np.sin(theta1_flipped) * radial - y) < tolerance)
            )
            theta1_selected = np.where(first_valid, theta1, theta1_flipped)

            inverse_10 = self._inverse_dh_rotations(theta1_selected, self.alpha[0])
            inverse_21 = self._inverse_dh_rotations(theta2, self.alpha[1])
            inverse_32 = self._inverse_dh_rotations(theta3, self.alpha[2])
            inverse_30 = inverse_32 @ inverse_21 @ inverse_10
            rotation_36 = inverse_30 @ rotations
            singular = (np.abs(rotation_36[:, 0, 2]) < 1e-5) & (
                np.abs(rotation_36[:, 2, 2]) < 1e-5
            )
            theta4 = np.where(
                singular,
                0.0,
                np.arctan2(rotation_36[:, 2, 2], -rotation_36[:, 0, 2]),
            )
            rotation_46 = self._inverse_dh_rotations(theta4, self.alpha[3]) @ rotation_36
            base_solution = np.stack(
                (
                    theta1_selected - self.theta_offset[0],
                    theta2 - self.theta_offset[1],
                    theta3 - self.theta_offset[2],
                    theta4 - self.theta_offset[3],
                    np.arctan2(-rotation_46[:, 0, 2], rotation_46[:, 2, 2]) - self.theta_offset[4],
                    np.arctan2(rotation_46[:, 1, 0], rotation_46[:, 1, 1]) - self.theta_offset[5],
                ),
                axis=1,
            )
            wrist_flipped = base_solution.copy()
            wrist_flipped[:, 3] += np.pi
            wrist_flipped[:, 4] = -base_solution[:, 4] - 2.0 * self.theta_offset[4]
            wrist_flipped[:, 5] += np.pi
            base_solution, base_limits_valid = self.joint_space.normalize(base_solution)
            wrist_flipped, flipped_limits_valid = self.joint_space.normalize(wrist_flipped)
            common_valid = workspace_valid & (second_discriminant >= -1e-9) & (first_valid | flipped_valid)

            base_branch = 2 * sign_index
            if base_branch in branch_output:
                output_index = branch_output[base_branch]
                joints[:, output_index] = base_solution
                valid[:, output_index] = common_valid & base_limits_valid
            if base_branch + 1 in branch_output:
                output_index = branch_output[base_branch + 1]
                joints[:, output_index] = wrist_flipped
                valid[:, output_index] = common_valid & flipped_limits_valid
        return joints, valid

    def forward_kinematics(self, joints: np.ndarray) -> ForwardKinematics:
        values = self._validate_joints(joints)
        physical = values + self.theta_offset[None, :]
        batch_size = len(values)
        links = np.broadcast_to(np.eye(4), (batch_size, 7, 4, 4)).copy()
        cumulative = links[:, 0].copy()
        for index in range(6):
            cumulative = cumulative @ self._forward_dh_transforms(
                self.a[index], self.alpha[index], self.d[index], physical[:, index]
            )
            links[:, index + 1] = cumulative
        tcp = links[:, -1].copy()
        tcp[:, :3, 3] += np.einsum("bij,j->bi", tcp[:, :3, :3], self.tool_offset)
        return ForwardKinematics(link_transforms=links, tcp_transforms=tcp)

    def inverse_dynamics(
        self,
        joints: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
        *,
        chassis_acceleration: np.ndarray | None = None,
        include_link_inertia: bool = False,
    ) -> np.ndarray:
        q = self._validate_joints(joints)
        dq = self._validate_joints(velocities, name="velocities")
        ddq = self._validate_joints(accelerations, name="accelerations")
        if dq.shape != q.shape or ddq.shape != q.shape:
            raise ValueError("joints, velocities, and accelerations must have identical shapes")
        chassis = np.zeros(len(q)) if chassis_acceleration is None else np.asarray(chassis_acceleration, dtype=float)
        if chassis.shape != (len(q),):
            raise ValueError(f"chassis_acceleration must have shape ({len(q)},), got {chassis.shape}")
        forward, inverse = self._dh_rotation_chains(q)
        translations = self._parent_translations()
        axis = np.asarray([0.0, 0.0, 1.0])
        angular_velocity = np.zeros((len(q), 7, 3))
        angular_acceleration = np.zeros((len(q), 7, 3))
        linear_acceleration = np.zeros((len(q), 7, 3))
        linear_acceleration[:, 0, 1] = -chassis
        linear_acceleration[:, 0, 2] = 9.81
        forces = np.zeros((len(q), 6, 3))
        moments = np.zeros((len(q), 6, 3))

        for index in range(6):
            rotated_velocity = np.einsum(
                "bij,bj->bi", inverse[:, index], angular_velocity[:, index]
            )
            joint_velocity = dq[:, index, None] * axis
            angular_velocity[:, index + 1] = rotated_velocity + joint_velocity
            angular_acceleration[:, index + 1] = (
                np.einsum(
                    "bij,bj->bi",
                    inverse[:, index],
                    angular_acceleration[:, index],
                )
                + np.cross(rotated_velocity, joint_velocity)
                + ddq[:, index, None] * axis
            )
            parent_acceleration = (
                np.cross(angular_acceleration[:, index], translations[index])
                + np.cross(
                    angular_velocity[:, index],
                    np.cross(angular_velocity[:, index], translations[index]),
                )
                + linear_acceleration[:, index]
            )
            linear_acceleration[:, index + 1] = np.einsum(
                "bij,bj->bi", inverse[:, index], parent_acceleration
            )
            center = self.centers_of_mass[index]
            center_acceleration = (
                np.cross(angular_acceleration[:, index + 1], center)
                + np.cross(
                    angular_velocity[:, index + 1],
                    np.cross(angular_velocity[:, index + 1], center),
                )
                + linear_acceleration[:, index + 1]
            )
            forces[:, index] = self.masses[index] * center_acceleration
            inertia_velocity = np.einsum(
                "ij,bj->bi", self.inertias[index], angular_velocity[:, index + 1]
            )
            moments[:, index] = np.einsum(
                "ij,bj->bi", self.inertias[index], angular_acceleration[:, index + 1]
            ) + np.cross(angular_velocity[:, index + 1], inertia_velocity)

        return self._rnea_backward(
            forward,
            forces,
            moments,
            np.zeros((len(q), 3)),
            np.zeros((len(q), 3)),
            include_link_inertia,
        )

    def external_wrench_torque(self, joints: np.ndarray, tcp_wrenches: np.ndarray) -> np.ndarray:
        q = self._validate_joints(joints)
        wrenches = np.asarray(tcp_wrenches, dtype=float)
        if wrenches.shape != (len(q), 6):
            raise ValueError(f"tcp_wrenches must have shape ({len(q)}, 6), got {wrenches.shape}")
        force = wrenches[:, :3]
        torque_frame6 = wrenches[:, 3:] + np.cross(self.tool_offset, force)
        forward, _ = self._dh_rotation_chains(q)
        zeros = np.zeros((len(q), 6, 3))
        return self._rnea_backward(forward, zeros, zeros, force, torque_frame6, False)

    def _validate_joints(self, joints: np.ndarray, *, name: str = "joints") -> np.ndarray:
        values = np.asarray(joints, dtype=float)
        if values.ndim != 2 or values.shape[1] != 6:
            raise ValueError(f"{name} must have shape (B, 6), got {values.shape}")
        return values

    @staticmethod
    def _normalize_branches(branches) -> tuple[int, ...]:
        values = tuple(range(8)) if branches is None else tuple(int(value) for value in branches)
        if not values or len(set(values)) != len(values) or any(value < 0 or value >= 8 for value in values):
            raise ValueError(f"IK branches must be unique values in [0, 7], got {values}")
        return values

    @staticmethod
    def _inverse_dh_rotations(theta: np.ndarray, alpha: float) -> np.ndarray:
        cosine = np.cos(theta)
        sine = np.sin(theta)
        cosine_alpha = np.cos(alpha)
        sine_alpha = np.sin(alpha)
        result = np.empty((len(theta), 3, 3), dtype=float)
        result[:, 0, 0] = cosine
        result[:, 0, 1] = sine * cosine_alpha
        result[:, 0, 2] = sine * sine_alpha
        result[:, 1, 0] = -sine
        result[:, 1, 1] = cosine * cosine_alpha
        result[:, 1, 2] = cosine * sine_alpha
        result[:, 2, 0] = 0.0
        result[:, 2, 1] = -sine_alpha
        result[:, 2, 2] = cosine_alpha
        return result

    @staticmethod
    def _forward_dh_transforms(a: float, alpha: float, d: float, theta: np.ndarray) -> np.ndarray:
        cosine = np.cos(theta)
        sine = np.sin(theta)
        cosine_alpha = np.cos(alpha)
        sine_alpha = np.sin(alpha)
        result = np.zeros((len(theta), 4, 4), dtype=float)
        result[:, 0, 0] = cosine
        result[:, 0, 1] = -sine
        result[:, 0, 3] = a
        result[:, 1, 0] = sine * cosine_alpha
        result[:, 1, 1] = cosine * cosine_alpha
        result[:, 1, 2] = -sine_alpha
        result[:, 1, 3] = -d * sine_alpha
        result[:, 2, 0] = sine * sine_alpha
        result[:, 2, 1] = cosine * sine_alpha
        result[:, 2, 2] = cosine_alpha
        result[:, 2, 3] = d * cosine_alpha
        result[:, 3, 3] = 1.0
        return result

    def _dh_rotation_chains(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = self._validate_joints(joints)
        forward = np.broadcast_to(np.eye(3), (len(values), 7, 3, 3)).copy()
        inverse = np.empty((len(values), 6, 3, 3))
        for index in range(6):
            theta = values[:, index] + self.theta_offset[index]
            transform = self._forward_dh_transforms(
                self.a[index], self.alpha[index], self.d[index], theta
            )
            forward[:, index] = transform[:, :3, :3]
            inverse[:, index] = np.swapaxes(transform[:, :3, :3], 1, 2)
        return forward, inverse

    def _parent_translations(self) -> np.ndarray:
        return np.stack(
            (
                self.a,
                -self.d * np.sin(self.alpha),
                self.d * np.cos(self.alpha),
            ),
            axis=1,
        )

    def _rnea_backward(self, forward, forces, moments, tip_force, tip_torque, include_link_inertia):
        translations = np.vstack((self._parent_translations(), np.zeros(3)))
        batch_size = len(forward)
        force = np.zeros((batch_size, 7, 3))
        torque = np.zeros((batch_size, 7, 3))
        force[:, 6] = np.asarray(tip_force, dtype=float)
        torque[:, 6] = np.asarray(tip_torque, dtype=float)
        joint_torque = np.zeros((batch_size, 6))
        for frame in range(6, 0, -1):
            propagated_force = np.einsum(
                "bij,bj->bi", forward[:, frame], force[:, frame]
            )
            force[:, frame - 1] = propagated_force + forces[:, frame - 1]
            torque[:, frame - 1] = (
                np.einsum("bij,bj->bi", forward[:, frame], torque[:, frame])
                + np.cross(translations[frame], propagated_force)
                + np.cross(self.centers_of_mass[frame - 1], forces[:, frame - 1])
            )
            if include_link_inertia:
                torque[:, frame - 1] += moments[:, frame - 1]
            joint_torque[:, frame - 1] = torque[:, frame - 1, 2]
        return joint_torque
