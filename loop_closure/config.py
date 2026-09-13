"""Configuration utilities shared by the loop-closure integration."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_HASH_CHUNK_SIZE = 1024 * 1024


def compute_file_sha256(
    path: Path, chunk_size: int = DEFAULT_HASH_CHUNK_SIZE
) -> str:
    """Return a file's SHA256 digest without loading the whole file in memory."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _require_int(value: Any, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _require_float(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    maximum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and result <= minimum:
        raise ValueError(f"{field} must be > {minimum}")
    if maximum is not None:
        invalid = result > maximum if maximum_inclusive else result >= maximum
        if invalid:
            operator = "<=" if maximum_inclusive else "<"
            raise ValueError(f"{field} must be {operator} {maximum}")
    return result


def _require_unit_interval(value: Any, field: str) -> float:
    result = _require_float(value, field)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return result


def _require_choice(value: Any, field: str, choices: set[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{field} must be one of: {allowed}")
    return value


def _resolve_asset(config_dir: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    path = path.resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"{field} does not reference a non-empty file: {path}")
    return path


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int
    temporal_exclusion: int


@dataclass(frozen=True)
class BoWGConfig:
    pattern_path: Path
    vocabulary_path: Path
    feature_count: int
    multi_scale: bool
    multi_scale_levels: int
    feature_size: float
    word_weight_type: int
    word_scoring_type: int
    word_group_weight_type: int
    word_group_scoring_type: int
    use_word_groups: bool
    use_distribution: bool
    word_weight: float
    word_group_weight: float
    use_temporal_score: bool
    previous_weight_threshold: float
    temporal_parameter: float
    similarity_threshold: float
    temporal_consistency: int
    max_intraisland_gap: int
    max_distance_between_groups: int
    max_distance_between_queries: int
    use_native_geometry: bool
    direct_index_geometry: bool
    direct_index_level: int
    native_min_feature_points: int
    native_max_reprojection_error: float
    native_ransac_probability: float
    native_ransac_max_iterations: int
    native_max_neighbor_ratio: float
    min_previous_word_score: float
    min_previous_word_group_score: float
    min_previous_distribution_score: float


@dataclass(frozen=True)
class GeometryConfig:
    depth_trunc_m: float
    depth_neighborhood_radius: int
    depth_discontinuity_m: float
    descriptor_ratio: float
    min_matches: int
    min_inliers: int
    min_inlier_ratio: float
    ransac_threshold_m: float
    ransac_probability: float
    ransac_max_iterations: int
    ransac_seed: int
    min_image_coverage: float
    coverage_grid_rows: int
    coverage_grid_cols: int
    min_singular_value_ratio: float
    max_residual_median_m: float
    max_residual_p95_m: float
    max_rotation_deg: float
    max_translation_m: float
    information_translation_weight: float
    information_rotation_weight: float
    use_gicp_refinement: bool


@dataclass(frozen=True)
class ISAM2Config:
    relinearize_threshold: float
    relinearize_skip: int
    factorization: str
    cache_linearized_factors: bool
    extra_update_steps: int


@dataclass(frozen=True)
class BackendConfig:
    enabled: bool
    optimize_every_n_loops: int
    huber_delta: float
    odometry_translation_weight: float
    odometry_rotation_weight: float
    loop_translation_weight: float
    loop_rotation_weight: float
    max_loop_chi2: float
    isam2: ISAM2Config


@dataclass(frozen=True)
class LoopClosureConfig:
    enabled: bool
    mode: str
    keyframe_source: str
    queue_capacity: int
    queue_policy: str
    worker_shutdown_timeout_s: float
    retrieval: RetrievalConfig
    bowg: BoWGConfig
    geometry: GeometryConfig
    backend: BackendConfig
    source_path: Path

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LoopClosureConfig":
        source_path = Path(path).expanduser().resolve()
        if not source_path.is_file():
            raise ValueError(f"config path does not exist: {source_path}")
        try:
            loaded = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid YAML config {source_path}: {exc}") from exc
        root = _require_mapping(loaded, "config")
        config_dir = source_path.parent

        retrieval_data = _require_mapping(root.get("retrieval"), "retrieval")
        retrieval = RetrievalConfig(
            top_k=_require_int(retrieval_data.get("top_k"), "retrieval.top_k", 1),
            temporal_exclusion=_require_int(
                retrieval_data.get("temporal_exclusion"),
                "retrieval.temporal_exclusion",
                1,
            ),
        )

        bowg_data = _require_mapping(root.get("bowg"), "bowg")
        bowg = BoWGConfig(
            pattern_path=_resolve_asset(
                config_dir, bowg_data.get("pattern_path"), "bowg.pattern_path"
            ),
            vocabulary_path=_resolve_asset(
                config_dir,
                bowg_data.get("vocabulary_path"),
                "bowg.vocabulary_path",
            ),
            feature_count=_require_int(
                bowg_data.get("feature_count"), "bowg.feature_count", 1
            ),
            multi_scale=_require_bool(bowg_data.get("multi_scale"), "bowg.multi_scale"),
            multi_scale_levels=_require_int(
                bowg_data.get("multi_scale_levels"), "bowg.multi_scale_levels", 1
            ),
            feature_size=_require_float(
                bowg_data.get("feature_size"), "bowg.feature_size", minimum=0.0
            ),
            word_weight_type=_require_int(
                bowg_data.get("word_weight_type"), "bowg.word_weight_type"
            ),
            word_scoring_type=_require_int(
                bowg_data.get("word_scoring_type"), "bowg.word_scoring_type"
            ),
            word_group_weight_type=_require_int(
                bowg_data.get("word_group_weight_type"),
                "bowg.word_group_weight_type",
            ),
            word_group_scoring_type=_require_int(
                bowg_data.get("word_group_scoring_type"),
                "bowg.word_group_scoring_type",
            ),
            use_word_groups=_require_bool(
                bowg_data.get("use_word_groups"), "bowg.use_word_groups"
            ),
            use_distribution=_require_bool(
                bowg_data.get("use_distribution"), "bowg.use_distribution"
            ),
            word_weight=_require_unit_interval(
                bowg_data.get("word_weight"), "bowg.word_weight"
            ),
            word_group_weight=_require_unit_interval(
                bowg_data.get("word_group_weight"),
                "bowg.word_group_weight",
            ),
            use_temporal_score=_require_bool(
                bowg_data.get("use_temporal_score"), "bowg.use_temporal_score"
            ),
            previous_weight_threshold=_require_float(
                bowg_data.get("previous_weight_threshold"),
                "bowg.previous_weight_threshold",
                minimum=0.0,
            ),
            temporal_parameter=_require_float(
                bowg_data.get("temporal_parameter"),
                "bowg.temporal_parameter",
                minimum=0.0,
            ),
            similarity_threshold=_require_unit_interval(
                bowg_data.get("similarity_threshold"),
                "bowg.similarity_threshold",
            ),
            temporal_consistency=_require_int(
                bowg_data.get("temporal_consistency"), "bowg.temporal_consistency"
            ),
            max_intraisland_gap=_require_int(
                bowg_data.get("max_intraisland_gap"),
                "bowg.max_intraisland_gap",
                1,
            ),
            max_distance_between_groups=_require_int(
                bowg_data.get("max_distance_between_groups"),
                "bowg.max_distance_between_groups",
                1,
            ),
            max_distance_between_queries=_require_int(
                bowg_data.get("max_distance_between_queries"),
                "bowg.max_distance_between_queries",
                1,
            ),
            use_native_geometry=_require_bool(
                bowg_data.get("use_native_geometry"), "bowg.use_native_geometry"
            ),
            direct_index_geometry=_require_bool(
                bowg_data.get("direct_index_geometry"),
                "bowg.direct_index_geometry",
            ),
            direct_index_level=_require_int(
                bowg_data.get("direct_index_level"), "bowg.direct_index_level", 1
            ),
            native_min_feature_points=_require_int(
                bowg_data.get("native_min_feature_points"),
                "bowg.native_min_feature_points",
                1,
            ),
            native_max_reprojection_error=_require_float(
                bowg_data.get("native_max_reprojection_error"),
                "bowg.native_max_reprojection_error",
                minimum=0.0,
            ),
            native_ransac_probability=_require_float(
                bowg_data.get("native_ransac_probability"),
                "bowg.native_ransac_probability",
                minimum=0.0,
                maximum=1.0,
            ),
            native_ransac_max_iterations=_require_int(
                bowg_data.get("native_ransac_max_iterations"),
                "bowg.native_ransac_max_iterations",
                1,
            ),
            native_max_neighbor_ratio=_require_float(
                bowg_data.get("native_max_neighbor_ratio"),
                "bowg.native_max_neighbor_ratio",
                minimum=0.0,
                maximum=1.0,
                maximum_inclusive=False,
            ),
            min_previous_word_score=_require_float(
                bowg_data.get("min_previous_word_score"),
                "bowg.min_previous_word_score",
                minimum=0.0,
            ),
            min_previous_word_group_score=_require_float(
                bowg_data.get("min_previous_word_group_score"),
                "bowg.min_previous_word_group_score",
                minimum=0.0,
            ),
            min_previous_distribution_score=_require_float(
                bowg_data.get("min_previous_distribution_score"),
                "bowg.min_previous_distribution_score",
                minimum=0.0,
            ),
        )

        geometry_data = _require_mapping(root.get("geometry"), "geometry")
        geometry = GeometryConfig(
            depth_trunc_m=_require_float(
                geometry_data.get("depth_trunc_m"),
                "geometry.depth_trunc_m",
                minimum=0.0,
            ),
            depth_neighborhood_radius=_require_int(
                geometry_data.get("depth_neighborhood_radius"),
                "geometry.depth_neighborhood_radius",
            ),
            depth_discontinuity_m=_require_float(
                geometry_data.get("depth_discontinuity_m"),
                "geometry.depth_discontinuity_m",
                minimum=0.0,
            ),
            descriptor_ratio=_require_float(
                geometry_data.get("descriptor_ratio"),
                "geometry.descriptor_ratio",
                minimum=0.0,
                maximum=1.0,
                maximum_inclusive=False,
            ),
            min_matches=_require_int(
                geometry_data.get("min_matches"), "geometry.min_matches", 1
            ),
            min_inliers=_require_int(
                geometry_data.get("min_inliers"), "geometry.min_inliers", 1
            ),
            min_inlier_ratio=_require_unit_interval(
                geometry_data.get("min_inlier_ratio"),
                "geometry.min_inlier_ratio",
            ),
            ransac_threshold_m=_require_float(
                geometry_data.get("ransac_threshold_m"),
                "geometry.ransac_threshold_m",
                minimum=0.0,
            ),
            ransac_probability=_require_float(
                geometry_data.get("ransac_probability"),
                "geometry.ransac_probability",
                minimum=0.0,
                maximum=1.0,
                maximum_inclusive=False,
            ),
            ransac_max_iterations=_require_int(
                geometry_data.get("ransac_max_iterations"),
                "geometry.ransac_max_iterations",
                1,
            ),
            ransac_seed=_require_int(
                geometry_data.get("ransac_seed"), "geometry.ransac_seed"
            ),
            min_image_coverage=_require_unit_interval(
                geometry_data.get("min_image_coverage"),
                "geometry.min_image_coverage",
            ),
            coverage_grid_rows=_require_int(
                geometry_data.get("coverage_grid_rows"),
                "geometry.coverage_grid_rows",
                1,
            ),
            coverage_grid_cols=_require_int(
                geometry_data.get("coverage_grid_cols"),
                "geometry.coverage_grid_cols",
                1,
            ),
            min_singular_value_ratio=_require_float(
                geometry_data.get("min_singular_value_ratio"),
                "geometry.min_singular_value_ratio",
                minimum=0.0,
                maximum=1.0,
            ),
            max_residual_median_m=_require_float(
                geometry_data.get("max_residual_median_m"),
                "geometry.max_residual_median_m",
                minimum=0.0,
            ),
            max_residual_p95_m=_require_float(
                geometry_data.get("max_residual_p95_m"),
                "geometry.max_residual_p95_m",
                minimum=0.0,
            ),
            max_rotation_deg=_require_float(
                geometry_data.get("max_rotation_deg"),
                "geometry.max_rotation_deg",
                minimum=0.0,
                maximum=180.0,
            ),
            max_translation_m=_require_float(
                geometry_data.get("max_translation_m"),
                "geometry.max_translation_m",
                minimum=0.0,
            ),
            information_translation_weight=_require_float(
                geometry_data.get("information_translation_weight"),
                "geometry.information_translation_weight",
                minimum=0.0,
            ),
            information_rotation_weight=_require_float(
                geometry_data.get("information_rotation_weight"),
                "geometry.information_rotation_weight",
                minimum=0.0,
            ),
            use_gicp_refinement=_require_bool(
                geometry_data.get("use_gicp_refinement"),
                "geometry.use_gicp_refinement",
            ),
        )
        if geometry.min_inliers > geometry.min_matches:
            raise ValueError("geometry.min_inliers must not exceed geometry.min_matches")
        if geometry.max_residual_p95_m < geometry.max_residual_median_m:
            raise ValueError(
                "geometry.max_residual_p95_m must not be below "
                "geometry.max_residual_median_m"
            )

        backend_data = _require_mapping(root.get("backend"), "backend")
        if "max_iterations" in backend_data:
            raise ValueError(
                "backend.max_iterations was removed with the SciPy optimizer"
            )
        isam2_data = _require_mapping(backend_data.get("isam2"), "backend.isam2")
        allowed_isam2_fields = {
            "relinearize_threshold",
            "relinearize_skip",
            "factorization",
            "cache_linearized_factors",
            "extra_update_steps",
        }
        unknown_isam2_fields = sorted(set(isam2_data) - allowed_isam2_fields)
        if unknown_isam2_fields:
            raise ValueError(
                f"unknown backend.isam2.{unknown_isam2_fields[0]} field"
            )
        isam2 = ISAM2Config(
            relinearize_threshold=_require_float(
                isam2_data.get("relinearize_threshold"),
                "backend.isam2.relinearize_threshold",
                minimum=0.0,
            ),
            relinearize_skip=_require_int(
                isam2_data.get("relinearize_skip"),
                "backend.isam2.relinearize_skip",
                1,
            ),
            factorization=_require_choice(
                isam2_data.get("factorization"),
                "backend.isam2.factorization",
                {"CHOLESKY", "QR"},
            ),
            cache_linearized_factors=_require_bool(
                isam2_data.get("cache_linearized_factors"),
                "backend.isam2.cache_linearized_factors",
            ),
            extra_update_steps=_require_int(
                isam2_data.get("extra_update_steps"),
                "backend.isam2.extra_update_steps",
                0,
            ),
        )
        backend = BackendConfig(
            enabled=_require_bool(backend_data.get("enabled"), "backend.enabled"),
            optimize_every_n_loops=_require_int(
                backend_data.get("optimize_every_n_loops"),
                "backend.optimize_every_n_loops",
                1,
            ),
            huber_delta=_require_float(
                backend_data.get("huber_delta"), "backend.huber_delta", minimum=0.0
            ),
            odometry_translation_weight=_require_float(
                backend_data.get("odometry_translation_weight"),
                "backend.odometry_translation_weight",
                minimum=0.0,
            ),
            odometry_rotation_weight=_require_float(
                backend_data.get("odometry_rotation_weight"),
                "backend.odometry_rotation_weight",
                minimum=0.0,
            ),
            loop_translation_weight=_require_float(
                backend_data.get("loop_translation_weight"),
                "backend.loop_translation_weight",
                minimum=0.0,
            ),
            loop_rotation_weight=_require_float(
                backend_data.get("loop_rotation_weight"),
                "backend.loop_rotation_weight",
                minimum=0.0,
            ),
            max_loop_chi2=_require_float(
                backend_data.get("max_loop_chi2"),
                "backend.max_loop_chi2",
                minimum=0.0,
            ),
            isam2=isam2,
        )

        return cls(
            enabled=_require_bool(root.get("enabled"), "enabled"),
            mode=_require_choice(
                root.get("mode"), "mode", {"log_only", "offline", "online"}
            ),
            keyframe_source=_require_choice(
                root.get("keyframe_source"), "keyframe_source", {"tracking"}
            ),
            queue_capacity=_require_int(
                root.get("queue_capacity"), "queue_capacity", 1
            ),
            queue_policy=_require_choice(
                root.get("queue_policy"), "queue_policy", {"block", "drop_newest"}
            ),
            worker_shutdown_timeout_s=_require_float(
                root.get("worker_shutdown_timeout_s"),
                "worker_shutdown_timeout_s",
                minimum=0.0,
            ),
            retrieval=retrieval,
            bowg=bowg,
            geometry=geometry,
            backend=backend,
            source_path=source_path,
        )

    def resolved_config(self) -> dict[str, Any]:
        """Return a JSON-compatible snapshot with resolved paths and asset hashes."""
        result = asdict(self)
        result["source_path"] = str(self.source_path)
        result["bowg"]["pattern_path"] = str(self.bowg.pattern_path)
        result["bowg"]["vocabulary_path"] = str(self.bowg.vocabulary_path)
        result["assets"] = {
            "pattern_sha256": compute_file_sha256(self.bowg.pattern_path),
            "vocabulary_sha256": compute_file_sha256(self.bowg.vocabulary_path),
        }
        json.dumps(result)
        return result
