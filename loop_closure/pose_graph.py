"""Deterministic, auditable offline SE(3) pose graph backend."""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .config import BackendConfig
from .types import LoopConstraint, validate_transform


_SMALL_ANGLE = 1e-8


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Map [translation, rotation-vector] tangent coordinates into SE(3)."""
    value = np.asarray(xi, dtype=np.float64)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError("xi must be a finite vector with shape (6,)")
    rho, omega = value[:3], value[3:]
    theta = float(np.linalg.norm(omega))
    omega_hat = _skew(omega)
    omega_hat2 = omega_hat @ omega_hat
    if theta < _SMALL_ANGLE:
        rotation = np.eye(3) + omega_hat + 0.5 * omega_hat2
        V = np.eye(3) + 0.5 * omega_hat + (1.0 / 6.0) * omega_hat2
    else:
        theta2 = theta * theta
        rotation = (
            np.eye(3)
            + (np.sin(theta) / theta) * omega_hat
            + ((1.0 - np.cos(theta)) / theta2) * omega_hat2
        )
        V = (
            np.eye(3)
            + ((1.0 - np.cos(theta)) / theta2) * omega_hat
            + ((theta - np.sin(theta)) / (theta2 * theta)) * omega_hat2
        )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = V @ rho
    return transform


def se3_log(transform: np.ndarray) -> np.ndarray:
    """Map an SE(3) transform into [translation, rotation-vector] coordinates."""
    value = np.asarray(transform)
    validate_transform(value, "transform")
    rotation = value[:3, :3]
    from scipy.spatial.transform import Rotation

    omega = Rotation.from_matrix(
        np.array(rotation, dtype=np.float64, copy=True)
    ).as_rotvec()
    theta = float(np.linalg.norm(omega))
    omega_hat = _skew(omega)
    if theta < _SMALL_ANGLE:
        V_inverse = np.eye(3) - 0.5 * omega_hat + (1.0 / 12.0) * (
            omega_hat @ omega_hat
        )
    else:
        theta2 = theta * theta
        coefficient = (
            1.0 - 0.5 * theta / np.tan(0.5 * theta)
        ) / theta2
        V_inverse = np.eye(3) - 0.5 * omega_hat + coefficient * (
            omega_hat @ omega_hat
        )
    rho = V_inverse @ value[:3, 3]
    return np.concatenate((rho, omega))


def edge_residual(
    T_WC_i: np.ndarray, T_WC_j: np.ndarray, Z_CiCj: np.ndarray
) -> np.ndarray:
    """Return Log(inv(Z) @ inv(T_i) @ T_j) in translation-rotation order."""
    return se3_log(np.linalg.inv(Z_CiCj) @ np.linalg.inv(T_WC_i) @ T_WC_j)


def _owned_transform(value: np.ndarray, name: str) -> np.ndarray:
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    validate_transform(result, name)
    result.setflags(write=False)
    return result


def _owned_information(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    if result.shape != (6, 6) or not np.isfinite(result).all():
        raise ValueError("information must be a finite 6x6 matrix")
    if not np.allclose(result, result.T, atol=1e-9):
        raise ValueError("information must be symmetric")
    if np.min(np.linalg.eigvalsh(result)) <= 0.0:
        raise ValueError("information must be positive definite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PoseGraphEdge:
    i: int
    j: int
    edge_type: str
    Z_CiCj: np.ndarray
    information: np.ndarray
    status: str
    chi2: float | None
    reason: str
    verification: Mapping[str, object] | None


@dataclass(frozen=True)
class LoopAdmission:
    status: str
    chi2: float
    edge_index: int


@dataclass(frozen=True)
class PoseGraphResult:
    optimized_T_WC: Mapping[int, np.ndarray]
    success: bool
    status: str
    iterations: int
    initial_cost: float
    final_cost: float
    initial_residual_norm: float
    final_residual_norm: float
    solver: str
    gtsam_version: str | None
    update_count: int
    update_ms: float
    estimate_ms: float
    variables_relinearized: int | None
    variables_reeliminated: int | None


def _require_gtsam():
    try:
        return importlib.import_module("gtsam")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "GTSAM 4.2 is required when loop-closure backend is enabled; "
            "install the verified CPython/ABI-compatible gtsam wheel"
        ) from exc


def _to_pose3(transform: np.ndarray):
    value = _owned_transform(transform, "transform")
    return _require_gtsam().Pose3(np.array(value, copy=True))


def _from_pose3(pose) -> np.ndarray:
    return _owned_transform(np.asarray(pose.matrix()), "gtsam Pose3")


def _information_to_gtsam(information: np.ndarray) -> np.ndarray:
    value = _owned_information(information)
    permutation = np.array([3, 4, 5, 0, 1, 2])
    return np.array(value[np.ix_(permutation, permutation)], copy=True)


def _noise_from_information(information: np.ndarray):
    return _require_gtsam().noiseModel.Gaussian.Information(
        _information_to_gtsam(information)
    )


class PoseGraphBackend:
    """Own immutable raw graph inputs and the last valid optimized pose mapping."""

    def __init__(self, config: BackendConfig):
        self.config = config
        self._gtsam = _require_gtsam()
        self._gtsam_version = getattr(self._gtsam, "__version__", None)
        if self._gtsam_version is None:
            try:
                from importlib.metadata import version

                self._gtsam_version = version("gtsam")
            except Exception:
                self._gtsam_version = None
        self._raw_poses: dict[int, np.ndarray] = {}
        self._current_poses: dict[int, np.ndarray] = {}
        self._edges: list[PoseGraphEdge] = []
        params = self._gtsam.ISAM2Params()
        params.setRelinearizeThreshold(config.isam2.relinearize_threshold)
        params.relinearizeSkip = config.isam2.relinearize_skip
        params.setFactorization(config.isam2.factorization)
        params.cacheLinearizedFactors = config.isam2.cache_linearized_factors
        self._isam = self._gtsam.ISAM2(params)
        self._pending_graph = self._gtsam.NonlinearFactorGraph()
        self._pending_values = self._gtsam.Values()
        self._all_graph = self._gtsam.NonlinearFactorGraph()
        self._committed_node_ids: set[int] = set()
        self._update_count = 0
        self._terminal_failure = False

    def add_node(self, node_id: int, T_WC_raw: np.ndarray) -> None:
        if isinstance(node_id, bool) or not isinstance(node_id, int) or node_id < 0:
            raise ValueError("node_id must be a non-negative integer")
        if node_id in self._raw_poses:
            raise ValueError(f"duplicate node_id: {node_id}")
        if node_id != len(self._raw_poses):
            raise ValueError(f"node_id must equal next ordered entry_id {len(self._raw_poses)}")
        pose = _owned_transform(T_WC_raw, "T_WC_raw")
        self._raw_poses[node_id] = pose
        self._current_poses[node_id] = pose
        self._pending_values.insert(node_id, _to_pose3(pose))
        if node_id == 0:
            prior_noise = self._gtsam.noiseModel.Diagonal.Sigmas(
                np.full(6, 1e-9, dtype=np.float64)
            )
            self._stage_factor(
                self._gtsam.PriorFactorPose3(node_id, _to_pose3(pose), prior_noise)
            )

    def add_odometry(
        self,
        i: int,
        j: int,
        Z_CiCj: np.ndarray,
        information: np.ndarray,
    ) -> None:
        self._validate_endpoints(i, j)
        if j != i + 1:
            raise ValueError("odometry edge endpoints must be adjacent ordered nodes")
        if any(
            edge.edge_type == "odometry" and edge.i == i and edge.j == j
            for edge in self._edges
        ):
            raise ValueError(f"duplicate odometry edge: {i}->{j}")
        measurement = _owned_transform(Z_CiCj, "Z_CiCj")
        edge = PoseGraphEdge(
            i=i,
            j=j,
            edge_type="odometry",
            Z_CiCj=measurement,
            information=_owned_information(information),
            status="active",
            chi2=None,
            reason="raw_adjacent_pose",
            verification=None,
        )
        self._edges.append(edge)
        self._stage_factor(
            self._gtsam.BetweenFactorPose3(
                i, j, _to_pose3(measurement), _noise_from_information(edge.information)
            )
        )
        self._current_poses[j] = _owned_transform(
            self._current_poses[i] @ measurement,
            f"current_T_WC[{j}]",
        )
        if j not in self._committed_node_ids:
            self._pending_values.update(j, _to_pose3(self._current_poses[j]))

    def add_loop(self, constraint: LoopConstraint) -> LoopAdmission:
        if constraint.verification_status != "accepted":
            raise ValueError("loop constraint verification_status must be accepted")
        self._validate_endpoints(constraint.id_old, constraint.id_cur)
        measurement = _owned_transform(constraint.T_ColdCcur, "T_ColdCcur")
        information = np.diag(
            [self.config.loop_translation_weight] * 3
            + [self.config.loop_rotation_weight] * 3
        ).astype(np.float64)
        information = _owned_information(information)
        innovation = edge_residual(
            self._current_poses[constraint.id_old],
            self._current_poses[constraint.id_cur],
            measurement,
        )
        chi2 = float(innovation @ information @ innovation)
        status = (
            "quarantined_graph_conflict"
            if chi2 > self.config.max_loop_chi2
            else "active"
        )
        reason = (
            "graph_innovation_chi2_exceeded"
            if status == "quarantined_graph_conflict"
            else "geometry_and_graph_consistent"
        )
        self._edges.append(
            PoseGraphEdge(
                i=constraint.id_old,
                j=constraint.id_cur,
                edge_type="loop",
                Z_CiCj=measurement,
                information=information,
                status=status,
                chi2=chi2,
                reason=reason,
                verification=_constraint_diagnostics(constraint),
            )
        )
        if status == "active":
            base_noise = _noise_from_information(information)
            robust_noise = self._gtsam.noiseModel.Robust.Create(
                self._gtsam.noiseModel.mEstimator.Huber.Create(
                    self.config.huber_delta
                ),
                base_noise,
            )
            self._stage_factor(
                self._gtsam.BetweenFactorPose3(
                    constraint.id_old,
                    constraint.id_cur,
                    _to_pose3(measurement),
                    robust_noise,
                )
            )
        return LoopAdmission(
            status=(
                "quarantined_graph_conflict"
                if status == "quarantined_graph_conflict"
                else "accepted"
            ),
            chi2=chi2,
            edge_index=len(self._edges) - 1,
        )

    def optimize(self) -> PoseGraphResult:
        base = {node_id: pose for node_id, pose in self._current_poses.items()}
        if self._terminal_failure:
            return _pose_graph_result(
                base,
                success=False, status="optimizer_terminal_failure",
                update_count=self._update_count,
                gtsam_version=self._gtsam_version,
            )
        if self._pending_values.size() == 0 and self._pending_graph.size() == 0:
            return _pose_graph_result(
                base, success=True, status="no_pending_updates",
                update_count=self._update_count,
                gtsam_version=self._gtsam_version,
            )
        initial_cost = self._graph_error(base)
        update_ms = 0.0
        estimate_ms = 0.0
        try:
            started = time.perf_counter()
            update_result = self._isam.update(
                self._pending_graph, self._pending_values
            )
            for _ in range(self.config.isam2.extra_update_steps):
                update_result = self._isam.update()
            update_ms = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            estimate = self._isam.calculateEstimate()
            estimate_ms = (time.perf_counter() - started) * 1000.0
            candidate = {
                node_id: _from_pose3(estimate.atPose3(node_id))
                for node_id in sorted(self._raw_poses)
            }
            if set(candidate) != set(self._raw_poses) or not all(
                _transform_is_valid(pose) for pose in candidate.values()
            ):
                raise ValueError("GTSAM returned an incomplete or invalid estimate")
            if not np.allclose(candidate[0], self._raw_poses[0], atol=1e-7):
                raise ValueError("GTSAM anchor node drifted")
        except Exception as exc:
            self._terminal_failure = True
            return _pose_graph_result(
                base, success=False,
                status=f"optimizer_exception:{type(exc).__name__}",
                initial_cost=initial_cost, final_cost=initial_cost,
                update_count=self._update_count, update_ms=update_ms,
                estimate_ms=estimate_ms,
                gtsam_version=self._gtsam_version,
            )
        candidate[0] = self._raw_poses[0]
        final_cost = self._graph_error(candidate)
        self._current_poses = {
            node_id: _owned_transform(pose, f"optimized_T_WC[{node_id}]")
            for node_id, pose in candidate.items()
        }
        self._committed_node_ids.update(self._raw_poses)
        self._pending_graph = self._gtsam.NonlinearFactorGraph()
        self._pending_values = self._gtsam.Values()
        self._update_count += 1
        return _pose_graph_result(
            self._current_poses,
            success=True, status="optimized", iterations=1,
            initial_cost=initial_cost, final_cost=final_cost,
            initial_residual_norm=float(np.sqrt(max(0.0, 2.0 * initial_cost))),
            final_residual_norm=float(np.sqrt(max(0.0, 2.0 * final_cost))),
            update_count=self._update_count, update_ms=update_ms,
            estimate_ms=estimate_ms,
            variables_relinearized=_optional_update_metric(
                update_result, "getVariablesRelinearized"
            ),
            variables_reeliminated=_optional_update_metric(
                update_result, "getVariablesReeliminated"
            ),
            gtsam_version=self._gtsam_version,
        )

    def _stage_factor(self, factor) -> None:
        self._pending_graph.add(factor)
        self._all_graph.add(factor)

    def _graph_error(self, poses: Mapping[int, np.ndarray]) -> float:
        if not poses or self._all_graph.size() == 0:
            return 0.0
        values = self._gtsam.Values()
        for node_id, pose in poses.items():
            values.insert(node_id, _to_pose3(pose))
        try:
            return float(self._all_graph.error(values))
        except Exception:
            return 0.0

    def edge_records(self) -> tuple[PoseGraphEdge, ...]:
        return tuple(self._edges)

    def raw_poses(self) -> Mapping[int, np.ndarray]:
        return MappingProxyType(dict(self._raw_poses))

    def current_poses(self) -> Mapping[int, np.ndarray]:
        return MappingProxyType(dict(self._current_poses))

    @staticmethod
    def _stack_residuals(
        poses: Mapping[int, np.ndarray], edges: list[PoseGraphEdge]
    ) -> np.ndarray:
        values = []
        for edge in edges:
            cholesky = np.linalg.cholesky(edge.information)
            values.append(
                cholesky.T
                @ edge_residual(poses[edge.i], poses[edge.j], edge.Z_CiCj)
            )
        return np.concatenate(values) if values else np.empty(0, dtype=np.float64)

    def _validate_endpoints(self, i: int, j: int) -> None:
        if i not in self._raw_poses or j not in self._raw_poses:
            raise ValueError(f"unknown graph node in edge {i}->{j}")
        if i >= j:
            raise ValueError("edge endpoints must satisfy i < j")


def _constraint_diagnostics(constraint: LoopConstraint) -> Mapping[str, object]:
    transform = _owned_transform(constraint.T_ColdCcur, "T_ColdCcur")
    information = _owned_information(constraint.information)
    return MappingProxyType(
        {
            "T_ColdCcur": transform,
            "information": information,
            "n_matches": constraint.n_matches,
            "n_inliers": constraint.n_inliers,
            "inlier_ratio": constraint.inlier_ratio,
            "residual_median_m": constraint.residual_median_m,
            "residual_p95_m": constraint.residual_p95_m,
            "image_coverage": constraint.image_coverage,
            "verification_status": constraint.verification_status,
        }
    )


def _transform_is_valid(transform: np.ndarray) -> bool:
    try:
        validate_transform(np.asarray(transform, dtype=np.float64), "transform")
    except ValueError:
        return False
    return True


def _huber_cost(residual: np.ndarray, scale: float) -> float:
    absolute = np.abs(residual)
    quadratic = absolute <= scale
    terms = np.where(
        quadratic,
        residual * residual,
        2.0 * scale * absolute - scale * scale,
    )
    return float(0.5 * terms.sum())


def _pose_graph_result(
    poses: Mapping[int, np.ndarray],
    *,
    success: bool,
    status: str,
    iterations: int = 0,
    initial_cost: float = 0.0,
    final_cost: float = 0.0,
    initial_residual_norm: float = 0.0,
    final_residual_norm: float = 0.0,
    update_count: int = 0,
    update_ms: float = 0.0,
    estimate_ms: float = 0.0,
    variables_relinearized: int | None = None,
    variables_reeliminated: int | None = None,
    gtsam_version: str | None = None,
) -> PoseGraphResult:
    owned = {
        node_id: _owned_transform(pose, f"optimized_T_WC[{node_id}]")
        for node_id, pose in poses.items()
    }
    return PoseGraphResult(
        optimized_T_WC=MappingProxyType(owned),
        success=success,
        status=status,
        iterations=iterations,
        initial_cost=initial_cost,
        final_cost=final_cost,
        initial_residual_norm=initial_residual_norm,
        final_residual_norm=final_residual_norm,
        solver="gtsam_isam2",
        gtsam_version=gtsam_version,
        update_count=update_count,
        update_ms=update_ms,
        estimate_ms=estimate_ms,
        variables_relinearized=variables_relinearized,
        variables_reeliminated=variables_reeliminated,
    )


def _optional_update_metric(result, method_name: str) -> int | None:
    method = getattr(result, method_name, None)
    if method is None:
        return None
    try:
        return int(method())
    except (TypeError, ValueError, RuntimeError):
        return None
