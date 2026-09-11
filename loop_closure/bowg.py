"""Validated Python wrapper around the native BoWG retrieval engine."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import LoopClosureConfig, compute_file_sha256
from .types import BoWGFeatures, RetrievalCandidate, RetrievalResult

try:
    from . import _bowg
except ImportError:
    import _bowg  # type: ignore[no-redef]


class BoWGEngine:
    """Own one ordered BoWG database and expose validated NumPy results."""

    def __init__(self, config_path: str | Path):
        try:
            self._config = LoopClosureConfig.from_yaml(config_path)
        except ValueError as exc:
            if "bowg.vocabulary_path" in str(exc):
                raise RuntimeError(str(exc)) from exc
            raise
        native_config = asdict(self._config.bowg)
        native_config["pattern_path"] = str(self._config.bowg.pattern_path)
        native_config["vocabulary_path"] = str(self._config.bowg.vocabulary_path)
        self._descriptor_config_hash = _descriptor_hash(native_config)
        self._native = _bowg.Engine(native_config)

    def query_then_add(
        self,
        image_bgr: np.ndarray,
        *,
        top_k: int,
        temporal_exclusion: int,
    ) -> RetrievalResult:
        if not isinstance(image_bgr, np.ndarray):
            raise ValueError("image_bgr must be a numpy array")
        if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3:
            raise ValueError("image_bgr must be uint8 with shape HxWx3")
        if image_bgr.shape[0] == 0 or image_bgr.shape[1] == 0 or image_bgr.shape[2] != 3:
            raise ValueError("image_bgr must be a non-empty uint8 HxWx3 array")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be an integer >= 1")
        if (
            isinstance(temporal_exclusion, bool)
            or not isinstance(temporal_exclusion, int)
            or temporal_exclusion < 1
        ):
            raise ValueError("temporal_exclusion must be an integer >= 1")

        image = np.ascontiguousarray(image_bgr)
        raw: dict[str, Any] = self._native.query_then_add(
            image, top_k, temporal_exclusion
        )
        entry_id = int(raw["entry_id"])
        return RetrievalResult(
            entry_id=entry_id,
            frame_id=entry_id,
            candidates=tuple(
                RetrievalCandidate(entry_id=int(identifier), score=float(score))
                for identifier, score in raw["candidates"]
            ),
            features=BoWGFeatures(
                keypoints_xy=np.asarray(raw["keypoints_xy"], dtype=np.float32),
                descriptors=np.asarray(raw["descriptors"], dtype=np.uint8),
            ),
            descriptor_config_hash=self._descriptor_config_hash,
            extract_ms=float(raw["extract_ms"]),
            query_ms=float(raw["query_ms"]),
        )

    def reset(self) -> None:
        self._native.reset()

    @property
    def size(self) -> int:
        return int(self._native.size)


def _descriptor_hash(config: dict[str, Any]) -> str:
    payload = dict(config)
    payload["pattern_sha256"] = compute_file_sha256(Path(payload.pop("pattern_path")))
    payload["vocabulary_sha256"] = compute_file_sha256(
        Path(payload.pop("vocabulary_path"))
    )
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
