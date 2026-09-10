"""Validated message contracts for the loop-closure pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np


_TRANSFORM_ATOL = 1e-9
_ROTATION_ATOL = 1e-6


def validate_transform(T: np.ndarray, name: str) -> None:
    """Validate a float64 homogeneous transform belonging to SE(3)."""
    if not isinstance(T, np.ndarray) or T.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 numpy array")
    if T.dtype != np.float64:
        raise ValueError(f"{name} must have dtype float64")
    if not np.isfinite(T).all():
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(T[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=_TRANSFORM_ATOL):
        raise ValueError(f"{name} must have homogeneous last row [0, 0, 0, 1]")

    rotation = T[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_ROTATION_ATOL):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=_ROTATION_ATOL):
        raise ValueError(f"{name} rotation determinant must be positive and equal to one")


def _owned_read_only_array(value: np.ndarray, dtype: np.dtype, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != dtype:
        raise ValueError(f"{name} must be a numpy array with dtype {np.dtype(dtype).name}")
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _non_negative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _finite_non_negative(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class LoopKeyframe:
    frame_id: int
    keyframe_id: int
    timestamp: float
    sequence_id: int
    image_bgr: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    T_WC_raw: np.ndarray
    anchor_id: int
    map_version: int

    def __post_init__(self) -> None:
        for name in ("frame_id", "keyframe_id", "sequence_id", "anchor_id", "map_version"):
            _non_negative_integer(getattr(self, name), name)
        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")

        image = _owned_read_only_array(self.image_bgr, np.dtype(np.uint8), "image_bgr")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image_bgr must have shape HxWx3")
        depth = _owned_read_only_array(self.depth_m, np.dtype(np.float32), "depth_m")
        if depth.ndim != 2 or depth.shape != image.shape[:2]:
            raise ValueError("depth_m must be float32 HxW and match image_bgr")
        if not np.isfinite(depth).all() or np.any(depth < 0):
            raise ValueError("depth_m must contain finite, non-negative meter values")

        intrinsic = _owned_read_only_array(self.K, np.dtype(np.float64), "K")
        if intrinsic.shape != (3, 3):
            raise ValueError("K must have shape 3x3")
        if not np.isfinite(intrinsic).all() or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
            raise ValueError("K must contain finite positive focal lengths")
        if not np.allclose(intrinsic[2], [0.0, 0.0, 1.0], atol=_TRANSFORM_ATOL):
            raise ValueError("K last row must equal [0, 0, 1]")

        pose = _owned_read_only_array(self.T_WC_raw, np.dtype(np.float64), "T_WC_raw")
        validate_transform(pose, "T_WC_raw")
        object.__setattr__(self, "image_bgr", image)
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "K", intrinsic)
        object.__setattr__(self, "T_WC_raw", pose)


@dataclass(frozen=True)
class BoWGFeatures:
    keypoints_xy: np.ndarray
    descriptors: np.ndarray

    def __post_init__(self) -> None:
        points = _owned_read_only_array(
            self.keypoints_xy, np.dtype(np.float32), "keypoints_xy"
        )
        descriptors = _owned_read_only_array(
            self.descriptors, np.dtype(np.uint8), "descriptors"
        )
        if points.ndim != 2 or points.shape[1:] != (2,) or not np.isfinite(points).all():
            raise ValueError("keypoints_xy must be finite float32 with shape Nx2")
        if descriptors.ndim != 2 or descriptors.shape[1:] != (32,):
            raise ValueError("descriptors must be uint8 with shape Nx32")
        if descriptors.shape[0] != points.shape[0]:
            raise ValueError("descriptors rows must match keypoints_xy rows")
        object.__setattr__(self, "keypoints_xy", points)
        object.__setattr__(self, "descriptors", descriptors)


@dataclass(frozen=True)
class RetrievalCandidate:
    entry_id: int
    score: float

    def __post_init__(self) -> None:
        _non_negative_integer(self.entry_id, "entry_id")
        _finite_non_negative(self.score, "score")


@dataclass(frozen=True)
class RetrievalResult:
    entry_id: int
    frame_id: int
    candidates: tuple[RetrievalCandidate, ...]
    features: BoWGFeatures
    descriptor_config_hash: str
    extract_ms: float
    query_ms: float

    def __post_init__(self) -> None:
        _non_negative_integer(self.entry_id, "entry_id")
        _non_negative_integer(self.frame_id, "frame_id")
        candidates = tuple(self.candidates)
        candidate_ids = [candidate.entry_id for candidate in candidates]
        scores = [candidate.score for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)) or scores != sorted(scores, reverse=True):
            raise ValueError("candidates must have unique IDs and descending scores")
        if not isinstance(self.features, BoWGFeatures):
            raise ValueError("features must be BoWGFeatures")
        if len(self.descriptor_config_hash) != 64 or any(
            char not in "0123456789abcdef" for char in self.descriptor_config_hash
        ):
            raise ValueError("descriptor_config_hash must be a lowercase SHA256 digest")
        _finite_non_negative(self.extract_ms, "extract_ms")
        _finite_non_negative(self.query_ms, "query_ms")
        object.__setattr__(self, "candidates", candidates)


@dataclass(frozen=True)
class LoopConstraint:
    id_old: int
    id_cur: int
    T_ColdCcur: np.ndarray
    information: np.ndarray
    n_matches: int
    n_inliers: int
    inlier_ratio: float
    residual_median_m: float
    residual_p95_m: float
    image_coverage: float
    verification_status: str

    def __post_init__(self) -> None:
        _non_negative_integer(self.id_old, "id_old")
        _non_negative_integer(self.id_cur, "id_cur")
        if self.id_old >= self.id_cur:
            raise ValueError("id_old must be less than id_cur")
        transform = _owned_read_only_array(
            self.T_ColdCcur, np.dtype(np.float64), "T_ColdCcur"
        )
        validate_transform(transform, "T_ColdCcur")
        information = _owned_read_only_array(
            self.information, np.dtype(np.float64), "information"
        )
        if information.shape != (6, 6) or not np.isfinite(information).all():
            raise ValueError("information must be a finite float64 6x6 matrix")
        if not np.allclose(information, information.T, atol=_ROTATION_ATOL):
            raise ValueError("information must be symmetric")
        if np.min(np.linalg.eigvalsh(information)) <= 0:
            raise ValueError("information must be positive definite")
        _non_negative_integer(self.n_matches, "n_matches")
        _non_negative_integer(self.n_inliers, "n_inliers")
        if self.n_inliers > self.n_matches:
            raise ValueError("n_inliers must not exceed n_matches")
        for name in ("inlier_ratio", "image_coverage"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        _finite_non_negative(self.residual_median_m, "residual_median_m")
        _finite_non_negative(self.residual_p95_m, "residual_p95_m")
        if self.residual_p95_m < self.residual_median_m:
            raise ValueError("residual_p95_m must not be below residual_median_m")
        if not self.verification_status:
            raise ValueError("verification_status must not be empty")
        object.__setattr__(self, "T_ColdCcur", transform)
        object.__setattr__(self, "information", information)


@dataclass(frozen=True)
class PoseUpdate:
    base_version: int
    new_version: int
    optimized_T_WC: Mapping[int, np.ndarray]
    anchor_corrections: Mapping[int, np.ndarray]

    def __post_init__(self) -> None:
        _non_negative_integer(self.base_version, "base_version")
        _non_negative_integer(self.new_version, "new_version")
        if self.new_version != self.base_version + 1:
            raise ValueError("new_version must equal base_version + 1")

        object.__setattr__(
            self,
            "optimized_T_WC",
            _validated_transform_mapping(self.optimized_T_WC, "optimized_T_WC"),
        )
        object.__setattr__(
            self,
            "anchor_corrections",
            _validated_transform_mapping(self.anchor_corrections, "anchor_corrections"),
        )


def _validated_transform_mapping(
    values: Mapping[int, np.ndarray], name: str
) -> Mapping[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for identifier, transform in values.items():
        _non_negative_integer(identifier, f"{name} key")
        owned = _owned_read_only_array(transform, np.dtype(np.float64), name)
        validate_transform(owned, f"{name}[{identifier}]")
        result[int(identifier)] = owned
    return MappingProxyType(result)

