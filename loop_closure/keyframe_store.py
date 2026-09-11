"""Ordered, immutable storage for BoWG keyframes and retrieval results."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .types import (
    BoWGFeatures,
    LoopKeyframe,
    RetrievalCandidate,
    RetrievalResult,
)


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StoredKeyframe:
    entry_id: int
    packet: LoopKeyframe
    retrieval: RetrievalResult


class KeyframeStore:
    """Maintain the one-to-one ordered mapping between entries and frames."""

    def __init__(self, log_path: str | Path | None = None):
        self._records: list[StoredKeyframe] = []
        self._entry_by_frame: dict[int, int] = {}
        self._last_sequence_id: int | None = None
        self._last_map_version: int | None = None
        self._log_path = Path(log_path).expanduser().resolve() if log_path else None

    def append(self, packet: LoopKeyframe, retrieval: RetrievalResult) -> int:
        expected_entry_id = len(self._records)
        if retrieval.entry_id != expected_entry_id:
            raise ValueError(
                f"retrieval.entry_id must equal next store entry_id "
                f"{expected_entry_id}, got {retrieval.entry_id}"
            )
        if retrieval.frame_id != packet.frame_id:
            raise ValueError("retrieval.frame_id must match packet.frame_id")
        if packet.frame_id in self._entry_by_frame:
            raise ValueError(f"duplicate frame_id: {packet.frame_id}")
        if (
            self._last_sequence_id is not None
            and packet.sequence_id <= self._last_sequence_id
        ):
            raise ValueError("sequence_id must be strictly increasing")
        if (
            self._last_map_version is not None
            and packet.map_version < self._last_map_version
        ):
            raise ValueError("map_version must not move backwards")

        stored = StoredKeyframe(
            entry_id=expected_entry_id,
            packet=_copy_packet(packet),
            retrieval=_copy_retrieval(retrieval),
        )
        if self._log_path is not None:
            self._append_log(stored)

        self._records.append(stored)
        self._entry_by_frame[packet.frame_id] = expected_entry_id
        self._last_sequence_id = packet.sequence_id
        self._last_map_version = packet.map_version
        return expected_entry_id

    def by_entry_id(self, entry_id: int) -> StoredKeyframe:
        if isinstance(entry_id, bool) or not isinstance(entry_id, int):
            raise KeyError(f"unknown entry_id: {entry_id}")
        try:
            return self._records[entry_id] if entry_id >= 0 else _raise_key(entry_id)
        except IndexError as exc:
            raise KeyError(f"unknown entry_id: {entry_id}") from exc

    def entry_id_for_frame(self, frame_id: int) -> int | None:
        return self._entry_by_frame.get(frame_id)

    def __len__(self) -> int:
        return len(self._records)

    def _append_log(self, stored: StoredKeyframe) -> None:
        assert self._log_path is not None
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(
            _record_to_json(stored), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        with self._log_path.open("a", encoding="utf-8") as stream:
            stream.write(serialized + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def _raise_key(entry_id: int) -> StoredKeyframe:
    raise KeyError(f"unknown entry_id: {entry_id}")


def _copy_packet(packet: LoopKeyframe) -> LoopKeyframe:
    return LoopKeyframe(
        frame_id=packet.frame_id,
        keyframe_id=packet.keyframe_id,
        timestamp=packet.timestamp,
        sequence_id=packet.sequence_id,
        image_bgr=packet.image_bgr,
        depth_m=packet.depth_m,
        K=packet.K,
        T_WC_raw=packet.T_WC_raw,
        anchor_id=packet.anchor_id,
        map_version=packet.map_version,
    )


def _copy_retrieval(retrieval: RetrievalResult) -> RetrievalResult:
    return RetrievalResult(
        entry_id=retrieval.entry_id,
        frame_id=retrieval.frame_id,
        candidates=tuple(
            RetrievalCandidate(candidate.entry_id, candidate.score)
            for candidate in retrieval.candidates
        ),
        features=BoWGFeatures(
            keypoints_xy=retrieval.features.keypoints_xy,
            descriptors=retrieval.features.descriptors,
        ),
        descriptor_config_hash=retrieval.descriptor_config_hash,
        extract_ms=retrieval.extract_ms,
        query_ms=retrieval.query_ms,
    )


def _record_to_json(stored: StoredKeyframe) -> dict[str, object]:
    packet = stored.packet
    retrieval = stored.retrieval
    return {
        "schema_version": SCHEMA_VERSION,
        "entry_id": stored.entry_id,
        "frame_id": packet.frame_id,
        "keyframe_id": packet.keyframe_id,
        "timestamp": packet.timestamp,
        "sequence_id": packet.sequence_id,
        "anchor_id": packet.anchor_id,
        "map_version": packet.map_version,
        "K": packet.K.tolist(),
        "T_WC_raw": packet.T_WC_raw.tolist(),
        "descriptor_config_hash": retrieval.descriptor_config_hash,
        "candidates": [
            {"entry_id": candidate.entry_id, "score": candidate.score}
            for candidate in retrieval.candidates
        ],
        "feature_count": len(retrieval.features.keypoints_xy),
        "extract_ms": retrieval.extract_ms,
        "query_ms": retrieval.query_ms,
    }
