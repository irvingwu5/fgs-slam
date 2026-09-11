"""Deterministic offline BoWG retrieval over a dataset-neutral JSONL manifest."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import yaml

from .config import LoopClosureConfig
from .keyframe_store import KeyframeStore, SCHEMA_VERSION
from .types import LoopKeyframe, RetrievalResult


class RetrievalEngine(Protocol):
    def query_then_add(
        self, image_bgr: np.ndarray, *, top_k: int, temporal_exclusion: int
    ) -> RetrievalResult: ...


@dataclass(frozen=True)
class ManifestEntry:
    frame_id: int
    keyframe_id: int
    timestamp: float
    sequence_id: int
    image_path: Path
    depth_path: Path
    depth_scale: float
    K: np.ndarray
    T_WC_raw: np.ndarray
    anchor_id: int
    map_version: int


def run_offline(
    config_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    ground_truth_path: str | Path | None = None,
    engine_factory: Callable[[str | Path], RetrievalEngine] | None = None,
) -> dict[str, Any]:
    """Run ordered retrieval and return aggregate evaluation metrics."""
    config = LoopClosureConfig.from_yaml(config_path)
    entries = _load_manifest(manifest_path)
    ground_truth = _load_ground_truth(ground_truth_path)

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    if engine_factory is None:
        from .bowg import BoWGEngine

        engine_factory = BoWGEngine
    engine = engine_factory(config.source_path)
    store = KeyframeStore(destination / "keyframes.jsonl")

    retrieval_path = destination / "retrieval.jsonl"
    timing_path = destination / "timing.csv"
    evaluations: list[dict[str, Any]] = []
    with retrieval_path.open("w", encoding="utf-8") as retrieval_stream, timing_path.open(
        "w", encoding="utf-8", newline=""
    ) as timing_stream:
        timing_writer = csv.writer(timing_stream)
        timing_writer.writerow(
            ["entry_id", "frame_id", "extract_ms", "query_ms", "total_ms"]
        )
        for entry in entries:
            packet = _load_packet(entry)
            native_result = engine.query_then_add(
                packet.image_bgr,
                top_k=config.retrieval.top_k,
                temporal_exclusion=config.retrieval.temporal_exclusion,
            )
            retrieval = replace(native_result, frame_id=packet.frame_id)
            candidate_rows = [
                {
                    "entry_id": candidate.entry_id,
                    "frame_id": store.by_entry_id(candidate.entry_id).packet.frame_id,
                    "score": candidate.score,
                }
                for candidate in retrieval.candidates
            ]
            evaluation = _evaluate_query(
                packet.frame_id, candidate_rows, ground_truth, config.retrieval.top_k
            )
            if evaluation is not None:
                evaluations.append(evaluation)

            store.append(packet, retrieval)
            record: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "entry_id": retrieval.entry_id,
                "frame_id": packet.frame_id,
                "sequence_id": packet.sequence_id,
                "candidates": candidate_rows,
                "descriptor_config_hash": retrieval.descriptor_config_hash,
                "evaluation": evaluation,
            }
            retrieval_stream.write(
                json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            )
            timing_writer.writerow(
                [
                    retrieval.entry_id,
                    packet.frame_id,
                    retrieval.extract_ms,
                    retrieval.query_ms,
                    retrieval.extract_ms + retrieval.query_ms,
                ]
            )

    metrics = _aggregate_metrics(evaluations, ground_truth is not None)
    _write_metadata(destination, config, metrics)
    return metrics


def _load_manifest(path: str | Path) -> list[ManifestEntry]:
    source = Path(path).expanduser().resolve()
    rows = _read_jsonl(source)
    entries = [_parse_manifest_row(row, source.parent, index + 1) for index, row in enumerate(rows)]
    sequence_ids = [entry.sequence_id for entry in entries]
    frame_ids = [entry.frame_id for entry in entries]
    if len(sequence_ids) != len(set(sequence_ids)):
        raise ValueError("manifest sequence_id values must be unique")
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("manifest frame_id values must be unique")
    return sorted(entries, key=lambda entry: entry.sequence_id)


def _parse_manifest_row(
    row: dict[str, Any], base_dir: Path, line_number: int
) -> ManifestEntry:
    required = {
        "frame_id",
        "keyframe_id",
        "timestamp",
        "sequence_id",
        "image_path",
        "depth_path",
        "depth_scale",
        "K",
        "T_WC_raw",
        "anchor_id",
        "map_version",
    }
    missing = sorted(required - row.keys())
    if missing:
        raise ValueError(f"manifest line {line_number} missing fields: {', '.join(missing)}")
    image_path = _resolve_data_path(base_dir, row["image_path"], "image_path")
    depth_path = _resolve_data_path(base_dir, row["depth_path"], "depth_path")
    depth_scale = float(row["depth_scale"])
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth_scale must be finite and positive")
    return ManifestEntry(
        frame_id=int(row["frame_id"]),
        keyframe_id=int(row["keyframe_id"]),
        timestamp=float(row["timestamp"]),
        sequence_id=int(row["sequence_id"]),
        image_path=image_path,
        depth_path=depth_path,
        depth_scale=depth_scale,
        K=np.asarray(row["K"], dtype=np.float64),
        T_WC_raw=np.asarray(row["T_WC_raw"], dtype=np.float64),
        anchor_id=int(row["anchor_id"]),
        map_version=int(row["map_version"]),
    )


def _load_packet(entry: ManifestEntry) -> LoopKeyframe:
    image = _load_array(entry.image_path, color=True)
    depth_raw = _load_array(entry.depth_path, color=False)
    if depth_raw.ndim != 2:
        raise ValueError(f"depth_path must contain a single-channel image: {entry.depth_path}")
    depth_m = np.asarray(depth_raw, dtype=np.float32) / entry.depth_scale
    return LoopKeyframe(
        frame_id=entry.frame_id,
        keyframe_id=entry.keyframe_id,
        timestamp=entry.timestamp,
        sequence_id=entry.sequence_id,
        image_bgr=np.asarray(image),
        depth_m=depth_m,
        K=entry.K,
        T_WC_raw=entry.T_WC_raw,
        anchor_id=entry.anchor_id,
        map_version=entry.map_version,
    )


def _load_array(path: Path, *, color: bool) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV Python is required for non-NPY manifest images") from exc
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_UNCHANGED
    value = cv2.imread(str(path), flag)
    if value is None:
        raise ValueError(f"failed to read manifest asset: {path}")
    return value


def _load_ground_truth(path: str | Path | None) -> dict[int, frozenset[int]] | None:
    if path is None:
        return None
    source = Path(path).expanduser().resolve()
    result: dict[int, frozenset[int]] = {}
    for line_number, row in enumerate(_read_jsonl(source), start=1):
        if "frame_id" not in row or "valid_history_frame_ids" not in row:
            raise ValueError(f"ground truth line {line_number} has missing fields")
        frame_id = int(row["frame_id"])
        if frame_id in result:
            raise ValueError(f"duplicate ground truth frame_id: {frame_id}")
        history = frozenset(int(value) for value in row["valid_history_frame_ids"])
        if frame_id in history:
            raise ValueError("ground truth cannot include the query frame itself")
        result[frame_id] = history
    return result


def _evaluate_query(
    frame_id: int,
    candidates: list[dict[str, Any]],
    ground_truth: dict[int, frozenset[int]] | None,
    top_k: int,
) -> dict[str, Any] | None:
    if ground_truth is None or frame_id not in ground_truth:
        return None
    relevant = ground_truth[frame_id]
    candidate_frames = [int(candidate["frame_id"]) for candidate in candidates]
    top1 = candidate_frames[:1]
    topk = candidate_frames[:top_k]
    top1_hits = sum(frame in relevant for frame in top1)
    topk_hits = sum(frame in relevant for frame in topk)
    first_rank = next(
        (rank for rank, candidate in enumerate(candidate_frames, start=1) if candidate in relevant),
        None,
    )
    denominator = len(relevant)
    return {
        "valid_history_frame_ids": sorted(relevant),
        "top1_precision": float(top1_hits),
        "top1_recall": top1_hits / denominator if denominator else 1.0,
        "topk_precision": topk_hits / len(topk) if topk else 0.0,
        "topk_recall": topk_hits / denominator if denominator else 1.0,
        "first_correct_rank": first_rank,
    }


def _aggregate_metrics(
    evaluations: list[dict[str, Any]], ground_truth_available: bool
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "ground_truth_available": ground_truth_available,
        "evaluated_queries": len(evaluations),
    }
    for name in ("top1_precision", "top1_recall", "topk_precision", "topk_recall"):
        metrics[name] = (
            sum(float(item[name]) for item in evaluations) / len(evaluations)
            if evaluations
            else None
        )
    return metrics


def _write_metadata(
    destination: Path, config: LoopClosureConfig, metrics: dict[str, Any]
) -> None:
    resolved = config.resolved_config()
    (destination / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8"
    )
    assets = resolved["assets"]
    (destination / "assets.sha256").write_text(
        f"{assets['pattern_sha256']}  {config.bowg.pattern_path}\n"
        f"{assets['vocabulary_sha256']}  {config.bowg.vocabulary_path}\n",
        encoding="utf-8",
    )
    (destination / "metrics.json").write_text(
        json.dumps(metrics, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"JSONL file does not exist: {path}")
    result = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
        result.append(value)
    if not result:
        raise ValueError(f"JSONL file has no records: {path}")
    return result


def _resolve_data_path(base_dir: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{field} does not exist: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ground-truth")
    arguments = parser.parse_args()
    metrics = run_offline(
        arguments.config,
        arguments.manifest,
        arguments.output_dir,
        ground_truth_path=arguments.ground_truth,
    )
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
