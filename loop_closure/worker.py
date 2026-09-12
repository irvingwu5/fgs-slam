"""Ordered online BoWG retrieval worker and bounded queue helpers."""

from __future__ import annotations

import queue
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

from .config import LoopClosureConfig
from .keyframe_store import KeyframeStore, StoredKeyframe
from .types import LoopConstraint, LoopKeyframe


_STOP_MESSAGE = "__FGS_SLAM_LOOP_WORKER_STOP__"


@dataclass(frozen=True)
class QueuedLoopKeyframe:
    packet: LoopKeyframe
    enqueued_monotonic: float
    enqueue_wait_ms: float
    queue_depth_at_enqueue: int | None


class LoopWorkerFailed(RuntimeError):
    """Raised by the parent after SLAM saved when the loop worker failed."""


def raise_for_loop_worker_failure(statuses: list[dict[str, object]]) -> None:
    """Convert a reported child failure into the parent's final non-zero error."""
    failure = next(
        (status for status in reversed(statuses) if status.get("status") == "FAILED"),
        None,
    )
    if failure is None:
        return
    exception_type = failure.get("exception_type", "Exception")
    message = failure.get("message", "loop worker failed")
    details = failure.get("traceback", "")
    raise LoopWorkerFailed(f"BoWG worker failed: {exception_type}: {message}\n{details}")


def record_unreported_worker_exit(loop_process, statuses, stop_event) -> bool:
    """Turn a native/bootstrap child death into a visible failure immediately."""
    if loop_process is None or loop_process.is_alive():
        return False
    if any(status.get("status") == "FAILED" for status in statuses):
        return False
    if loop_process.exitcode == 0 and any(
        status.get("status") == "STOPPED" for status in statuses
    ):
        return False
    stop_event.set()
    statuses.append(
        {
            "status": "FAILED",
            "exception_type": "ChildProcessError",
            "message": (
                f"loop worker exited with code {loop_process.exitcode} "
                "without a terminal status"
            ),
            "traceback": "",
        }
    )
    return True


def stop_processes_after_slam_child_failure(processes, stop_event) -> None:
    """Prevent surviving Mapper/loop children from waiting after native SLAM death."""
    if stop_event is not None:
        stop_event.set()
    for process in processes:
        if process is not None and process.is_alive():
            process.terminate()
        if process is not None:
            process.join()


def worker_shutdown_stalled(
    last_progress_monotonic: float,
    timeout_s: float,
    *,
    now: float | None = None,
) -> bool:
    """Return true when shutdown has produced no observable progress in time."""
    current = time.monotonic() if now is None else now
    return current - last_progress_monotonic >= timeout_s


def build_tracking_keyframe(
    *,
    frame_id: int,
    keyframe_id: int,
    timestamp: float,
    image_bgr: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    T_WC_raw: np.ndarray,
) -> LoopKeyframe:
    """Build the immutable Task 5 packet for one tracking keyframe."""
    return LoopKeyframe(
        frame_id=frame_id,
        keyframe_id=keyframe_id,
        timestamp=timestamp,
        sequence_id=frame_id,
        image_bgr=image_bgr,
        depth_m=depth_m,
        K=np.asarray(K, dtype=np.float64),
        T_WC_raw=np.asarray(T_WC_raw, dtype=np.float64),
        anchor_id=keyframe_id,
        map_version=0,
    )


def publish_loop_keyframe(
    input_queue,
    packet: LoopKeyframe,
    stop_event,
    *,
    poll_interval_s: float = 0.05,
) -> dict[str, float | int | None] | None:
    """Put one packet without dropping it, unless worker failure cancels input."""
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be positive")
    started = time.monotonic()
    while not stop_event.is_set():
        queued_at = time.monotonic()
        queue_depth = _safe_qsize(input_queue)
        message = QueuedLoopKeyframe(
            packet=packet,
            enqueued_monotonic=started,
            enqueue_wait_ms=(queued_at - started) * 1000.0,
            queue_depth_at_enqueue=queue_depth,
        )
        try:
            input_queue.put_nowait(message)
            return {
                "enqueue_wait_ms": message.enqueue_wait_ms,
                "queue_depth": queue_depth,
            }
        except queue.Full:
            stop_event.wait(poll_interval_s)
    _cancel_queue_feeder(input_queue)
    return None


def request_loop_worker_stop(
    input_queue,
    stop_event,
    *,
    poll_interval_s: float = 0.05,
) -> bool:
    """Place the FIFO stop boundary unless worker failure already cancelled input."""
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be positive")
    while not stop_event.is_set():
        try:
            input_queue.put(_STOP_MESSAGE, timeout=poll_interval_s)
            return True
        except queue.Full:
            continue
    _cancel_queue_feeder(input_queue)
    return False


def loop_worker_main(
    config_path: str | Path,
    input_queue,
    output_queue,
    status_queue,
    stop_event,
    *,
    engine_factory: Callable | None = None,
    verifier_factory: Callable | None = None,
    log_path: str | Path | None = None,
) -> None:
    """Own one BoWG engine and process accepted keyframes in FIFO order."""
    processed = 0
    try:
        config = LoopClosureConfig.from_yaml(config_path)
        if engine_factory is None:
            from .bowg import BoWGEngine

            engine_factory = BoWGEngine
        if verifier_factory is None:
            from .geometric_verifier import RGBDGeometricVerifier

            verifier_factory = RGBDGeometricVerifier
        engine = engine_factory(config.source_path)
        verifier = verifier_factory(config.geometry)
        store = KeyframeStore(log_path)
        status_queue.put({"status": "STARTED"})

        while True:
            message = input_queue.get()
            if message == _STOP_MESSAGE:
                break
            if not isinstance(message, QueuedLoopKeyframe):
                raise TypeError(
                    "loop worker input must be QueuedLoopKeyframe or the stop message"
                )
            packet = message.packet
            native_result = engine.query_then_add(
                packet.image_bgr,
                top_k=config.retrieval.top_k,
                temporal_exclusion=config.retrieval.temporal_exclusion,
            )
            retrieval = replace(native_result, frame_id=packet.frame_id)
            current = StoredKeyframe(retrieval.entry_id, packet, retrieval)
            verifications: list[dict[str, object]] = []
            accepted_constraint: dict[str, object] | None = None
            for candidate in retrieval.candidates:
                old = store.by_entry_id(candidate.entry_id)
                constraint = verifier.verify(old, current, candidate.score)
                serialized = _constraint_to_dict(constraint, candidate.score)
                verifications.append(serialized)
                if constraint.verification_status == "accepted":
                    accepted_constraint = serialized
                    break
            store.append(packet, retrieval)
            processed += 1
            output_queue.put(
                {
                    "entry_id": retrieval.entry_id,
                    "frame_id": packet.frame_id,
                    "sequence_id": packet.sequence_id,
                    "candidates": [
                        {"entry_id": item.entry_id, "score": item.score}
                        for item in retrieval.candidates
                    ],
                    "verifications": verifications,
                    "accepted_constraint": accepted_constraint,
                    "descriptor_config_hash": retrieval.descriptor_config_hash,
                    "extract_ms": retrieval.extract_ms,
                    "query_ms": retrieval.query_ms,
                    "enqueue_wait_ms": message.enqueue_wait_ms,
                    "queue_depth_at_enqueue": message.queue_depth_at_enqueue,
                    "worker_lag_ms":
                        (time.monotonic() - message.enqueued_monotonic) * 1000.0,
                }
            )
        status_queue.put({"status": "STOPPED", "processed": processed})
    except BaseException as exc:
        stop_event.set()
        status_queue.put(
            {
                "status": "FAILED",
                "processed": processed,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _constraint_to_dict(
    constraint: LoopConstraint, score: float
) -> dict[str, object]:
    return {
        "id_old": constraint.id_old,
        "id_cur": constraint.id_cur,
        "bowg_score": float(score),
        "T_ColdCcur": constraint.T_ColdCcur.tolist(),
        "information": constraint.information.tolist(),
        "n_matches": constraint.n_matches,
        "n_inliers": constraint.n_inliers,
        "inlier_ratio": constraint.inlier_ratio,
        "residual_median_m": constraint.residual_median_m,
        "residual_p95_m": constraint.residual_p95_m,
        "image_coverage": constraint.image_coverage,
        "verification_status": constraint.verification_status,
    }


def _safe_qsize(value) -> int | None:
    try:
        return int(value.qsize())
    except (AttributeError, NotImplementedError):
        return None


def _cancel_queue_feeder(value) -> None:
    try:
        value.cancel_join_thread()
    except AttributeError:
        pass
