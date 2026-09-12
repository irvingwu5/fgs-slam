"""Deterministic RGB-D verification for ranked BoWG loop candidates."""

from __future__ import annotations

import math

import numpy as np

from .config import GeometryConfig
from .keyframe_store import StoredKeyframe
from .types import LoopConstraint


_POPCOUNT = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None], axis=1
).sum(axis=1).astype(np.uint8)
_HAMMING_BLOCK_ROWS = 128


class RGBDGeometricVerifier:
    """Estimate a metric current-to-old camera transform from two RGB-D frames."""

    def __init__(self, config: GeometryConfig):
        self.config = config

    def verify(
        self, old: StoredKeyframe, cur: StoredKeyframe, score: float
    ) -> LoopConstraint:
        if old.entry_id >= cur.entry_id:
            raise ValueError("old.entry_id must be less than cur.entry_id")
        if not np.isfinite(score) or score < 0:
            raise ValueError("score must be finite and non-negative")

        old_indices, cur_indices = _mutual_ratio_matches(
            old.retrieval.features.descriptors,
            cur.retrieval.features.descriptors,
            self.config.descriptor_ratio,
        )
        n_matches = len(old_indices)
        if n_matches < self.config.min_matches:
            return self._rejected(old, cur, "insufficient_matches", n_matches)

        old_points = []
        cur_points = []
        old_pixels = []
        cur_pixels = []
        for old_index, cur_index in zip(old_indices, cur_indices):
            old_pixel = old.retrieval.features.keypoints_xy[old_index]
            cur_pixel = cur.retrieval.features.keypoints_xy[cur_index]
            old_depth = _sample_depth(old.packet.depth_m, old_pixel, self.config)
            cur_depth = _sample_depth(cur.packet.depth_m, cur_pixel, self.config)
            if old_depth is None or cur_depth is None:
                continue
            old_points.append(_back_project(old_pixel, old_depth, old.packet.K))
            cur_points.append(_back_project(cur_pixel, cur_depth, cur.packet.K))
            old_pixels.append(old_pixel)
            cur_pixels.append(cur_pixel)

        if len(old_points) < self.config.min_matches:
            return self._rejected(
                old, cur, "insufficient_valid_depth", n_matches
            )
        old_xyz = np.asarray(old_points, dtype=np.float64)
        cur_xyz = np.asarray(cur_points, dtype=np.float64)
        if _is_degenerate(
            old_xyz, self.config.min_singular_value_ratio
        ) or _is_degenerate(cur_xyz, self.config.min_singular_value_ratio):
            return self._rejected(old, cur, "degenerate_geometry", n_matches)

        transform, inliers = _ransac_transform(
            cur_xyz,
            old_xyz,
            threshold=self.config.ransac_threshold_m,
            probability=self.config.ransac_probability,
            max_iterations=self.config.ransac_max_iterations,
            seed=self.config.ransac_seed + 1000003 * old.entry_id + cur.entry_id,
        )
        n_inliers = int(inliers.sum())
        inlier_ratio = n_inliers / len(old_xyz)
        if transform is None or n_inliers < self.config.min_inliers:
            return self._rejected(
                old,
                cur,
                "insufficient_inliers",
                n_matches,
                n_inliers=n_inliers,
                inlier_ratio=inlier_ratio,
            )
        if inlier_ratio < self.config.min_inlier_ratio:
            return self._rejected(
                old,
                cur,
                "low_inlier_ratio",
                n_matches,
                n_inliers=n_inliers,
                inlier_ratio=inlier_ratio,
            )

        transform = _estimate_rigid(cur_xyz[inliers], old_xyz[inliers])
        residuals = _residuals(transform, cur_xyz[inliers], old_xyz[inliers])
        if self.config.use_gicp_refinement:
            transform, residuals = _refine_with_gicp(
                transform, cur_xyz[inliers], old_xyz[inliers], residuals
            )
        median = float(np.median(residuals))
        p95 = float(np.percentile(residuals, 95))
        old_pixels_array = np.asarray(old_pixels)[inliers]
        cur_pixels_array = np.asarray(cur_pixels)[inliers]
        coverage = min(
            _image_coverage(
                old_pixels_array,
                old.packet.image_bgr.shape[:2],
                self.config.coverage_grid_rows,
                self.config.coverage_grid_cols,
            ),
            _image_coverage(
                cur_pixels_array,
                cur.packet.image_bgr.shape[:2],
                self.config.coverage_grid_rows,
                self.config.coverage_grid_cols,
            ),
        )
        if (
            median > self.config.max_residual_median_m
            or p95 > self.config.max_residual_p95_m
        ):
            status = "residual_too_large"
        elif coverage < self.config.min_image_coverage:
            status = "low_image_coverage"
        elif (
            _rotation_angle_deg(transform[:3, :3]) > self.config.max_rotation_deg
            or np.linalg.norm(transform[:3, 3]) > self.config.max_translation_m
        ):
            status = "motion_out_of_range"
        else:
            status = "accepted"
        return self._constraint(
            old,
            cur,
            transform,
            status,
            n_matches,
            n_inliers,
            inlier_ratio,
            median,
            p95,
            coverage,
        )

    def _rejected(
        self,
        old: StoredKeyframe,
        cur: StoredKeyframe,
        status: str,
        n_matches: int,
        *,
        n_inliers: int = 0,
        inlier_ratio: float = 0.0,
    ) -> LoopConstraint:
        return self._constraint(
            old,
            cur,
            np.eye(4, dtype=np.float64),
            status,
            n_matches,
            n_inliers,
            inlier_ratio,
            0.0,
            0.0,
            0.0,
            information=np.eye(6, dtype=np.float64),
        )

    def _constraint(
        self,
        old: StoredKeyframe,
        cur: StoredKeyframe,
        transform: np.ndarray,
        status: str,
        n_matches: int,
        n_inliers: int,
        inlier_ratio: float,
        median: float,
        p95: float,
        coverage: float,
        *,
        information: np.ndarray | None = None,
    ) -> LoopConstraint:
        if information is None:
            information = np.diag(
                [self.config.information_translation_weight] * 3
                + [self.config.information_rotation_weight] * 3
            ).astype(np.float64)
        return LoopConstraint(
            id_old=old.entry_id,
            id_cur=cur.entry_id,
            T_ColdCcur=np.asarray(transform, dtype=np.float64),
            information=information,
            n_matches=int(n_matches),
            n_inliers=int(n_inliers),
            inlier_ratio=float(inlier_ratio),
            residual_median_m=float(median),
            residual_p95_m=float(p95),
            image_coverage=float(coverage),
            verification_status=status,
        )


def _mutual_ratio_matches(
    old_descriptors: np.ndarray,
    cur_descriptors: np.ndarray,
    ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(old_descriptors) < 2 or len(cur_descriptors) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    distances = np.empty(
        (len(cur_descriptors), len(old_descriptors)), dtype=np.uint16
    )
    for start in range(0, len(cur_descriptors), _HAMMING_BLOCK_ROWS):
        stop = min(start + _HAMMING_BLOCK_ROWS, len(cur_descriptors))
        xor = np.bitwise_xor(
            cur_descriptors[start:stop, None, :], old_descriptors[None, :, :]
        )
        distances[start:stop] = _POPCOUNT[xor].sum(axis=2, dtype=np.uint16)
    nearest_two = np.argpartition(distances, kth=1, axis=1)[:, :2]
    nearest_two_distances = np.take_along_axis(distances, nearest_two, axis=1)
    order = np.argsort(nearest_two_distances, axis=1)
    nearest_two = np.take_along_axis(nearest_two, order, axis=1)
    nearest_two_distances = np.take_along_axis(nearest_two_distances, order, axis=1)
    best_old = nearest_two[:, 0]
    ratio_ok = nearest_two_distances[:, 0] < ratio * nearest_two_distances[:, 1]
    best_cur_for_old = np.argmin(distances, axis=0)
    cur_indices = np.arange(len(cur_descriptors), dtype=np.int64)
    mutual = best_cur_for_old[best_old] == cur_indices
    keep = ratio_ok & mutual
    return best_old[keep].astype(np.int64), cur_indices[keep]


def _sample_depth(
    depth_m: np.ndarray, pixel_xy: np.ndarray, config: GeometryConfig
) -> float | None:
    u, v = np.rint(pixel_xy).astype(int)
    radius = config.depth_neighborhood_radius
    if u < 0 or v < 0 or u >= depth_m.shape[1] or v >= depth_m.shape[0]:
        return None
    u0, u1 = max(0, u - radius), min(depth_m.shape[1], u + radius + 1)
    v0, v1 = max(0, v - radius), min(depth_m.shape[0], v + radius + 1)
    values = np.asarray(depth_m[v0:v1, u0:u1], dtype=np.float64).ravel()
    values = values[
        np.isfinite(values) & (values > 0.0) & (values <= config.depth_trunc_m)
    ]
    if len(values) == 0 or float(values.max() - values.min()) > config.depth_discontinuity_m:
        return None
    return float(np.median(values))


def _back_project(pixel_xy: np.ndarray, depth: float, K: np.ndarray) -> np.ndarray:
    u, v = pixel_xy
    return np.array(
        [
            (u - K[0, 2]) * depth / K[0, 0],
            (v - K[1, 2]) * depth / K[1, 1],
            depth,
        ],
        dtype=np.float64,
    )


def _is_degenerate(points: np.ndarray, minimum_ratio: float) -> bool:
    singular_values = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    return singular_values[0] <= 0 or singular_values[1] / singular_values[0] < minimum_ratio


def _sample_is_degenerate(points: np.ndarray) -> bool:
    """Return whether a minimal three-point sample is collinear."""
    singular_values = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    if singular_values[0] <= 0:
        return True
    return bool(singular_values[1] / singular_values[0] < 1e-8)


def _estimate_rigid(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    u, _, vt = np.linalg.svd(source_zero.T @ target_zero)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def _ransac_transform(
    source: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float,
    probability: float,
    max_iterations: int,
    seed: int,
) -> tuple[np.ndarray | None, np.ndarray]:
    rng = np.random.default_rng(seed)
    best_transform = None
    best_inliers = np.zeros(len(source), dtype=bool)
    best_median = math.inf
    required_iterations = max_iterations
    iteration = 0
    while iteration < min(max_iterations, required_iterations):
        indices = rng.choice(len(source), size=3, replace=False)
        sample_source = source[indices]
        sample_target = target[indices]
        if _sample_is_degenerate(sample_source) or _sample_is_degenerate(sample_target):
            iteration += 1
            continue
        transform = _estimate_rigid(sample_source, sample_target)
        residuals = _residuals(transform, source, target)
        inliers = residuals <= threshold
        count = int(inliers.sum())
        median = float(np.median(residuals[inliers])) if count else math.inf
        if count > int(best_inliers.sum()) or (
            count == int(best_inliers.sum()) and median < best_median
        ):
            best_transform = transform
            best_inliers = inliers
            best_median = median
            inlier_probability = count / len(source)
            failure_probability = 1.0 - inlier_probability ** 3
            if 0.0 < failure_probability < 1.0:
                required_iterations = min(
                    required_iterations,
                    max(
                        1,
                        math.ceil(
                            math.log(1.0 - probability)
                            / math.log(failure_probability)
                        ),
                    ),
                )
        iteration += 1
    return best_transform, best_inliers


def _residuals(transform: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    predicted = (transform[:3, :3] @ source.T).T + transform[:3, 3]
    return np.linalg.norm(predicted - target, axis=1)


def _image_coverage(
    pixels: np.ndarray,
    image_shape: tuple[int, int],
    rows: int,
    cols: int,
) -> float:
    if len(pixels) == 0:
        return 0.0
    height, width = image_shape
    col_ids = np.clip((pixels[:, 0] * cols / width).astype(int), 0, cols - 1)
    row_ids = np.clip((pixels[:, 1] * rows / height).astype(int), 0, rows - 1)
    occupied = np.unique(row_ids * cols + col_ids)
    return float(len(occupied) / (rows * cols))


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def _refine_with_gicp(
    transform: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    original_residuals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pygicp
    except ImportError as exc:
        raise RuntimeError("pygicp is required when use_gicp_refinement=true") from exc
    registration = pygicp.FastGICP()
    registration.set_input_source(source)
    registration.set_input_target(target)
    refined = np.asarray(registration.align(transform), dtype=np.float64)
    if refined.shape != (4, 4) or not np.isfinite(refined).all():
        return transform, original_residuals
    refined_residuals = _residuals(refined, source, target)
    if np.median(refined_residuals) <= np.median(original_residuals):
        return refined, refined_residuals
    return transform, original_residuals
