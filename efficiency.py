"""Edge-friendly runtime primitives shared by the YOLO segmentation pipeline.

Every object-recognition decision in this project is delegated to a trained
Ultralytics YOLO segmentation checkpoint.  Nothing here performs contour
extraction, background subtraction, colour thresholding or motion gating: the
helpers below only bound memory, bound CPU work and log state transitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
import json
import time

import cv2
import numpy as np


OPTIMIZATION_NAMES: tuple[str, ...] = (
    "one shared YOLO26 segmentation checkpoint per experiment",
    "native instance masks instead of box heuristics",
    "confidence and IoU gated inference",
    "bounding box area filter for small dark objects",
    "fixed ROI cropping for the gesture models",
    "consecutive-frame confirmation before a step passes",
    "rule-based ordered step state machine",
    "pre-baked multilingual audio phrases",
    "wake-word audio spotting",
    "event-driven logging",
    "capture and inference isolated on worker threads",
)


@dataclass(slots=True)
class EdgeTuning:
    """Tuning defaults selected for a small CPU-only edge device."""

    fixed_roi_fraction: float = 0.82
    confidence: float = 0.50
    iou: float = 0.50
    imgsz: int = 640
    #: Consecutive inference results that must all satisfy a step before it
    #: passes. 1 means a step advances on the first frame where every required
    #: label is present. Raise it to 2-3 to reject isolated single-frame
    #: detections at the cost of needing the objects to stay in view longer.
    match_frames: int = 1
    #: Gesture frames required before recording toggles.
    gesture_frames: int = 25
    gesture_cooldown_seconds: float = 3.0
    #: Small-object re-label rule. A black earbud case and a small black box are
    #: both dark plastic, so the checkpoint reports the case as ``blackbox``.
    #: Any detection of the source class whose bounding box area falls below the
    #: threshold is re-assigned the earbud class name. Set the threshold to 0
    #: (or leave the target empty) to disable the rule entirely.
    small_object_source_label: str = "blackbox"
    small_object_target_label: str = "black-earbuds"
    small_object_max_area: float = 25000.0
    #: Consecutive inference results a removal step's target and container must
    #: stay physically apart before the step passes.
    separation_frames: int = 10
    #: Overlap at or above this IoU means the two boxes are still touching.
    separation_max_iou: float = 0.15
    #: Overlap at or above this fraction of the smaller box means one object is
    #: inside the other, which counts as "not separated" regardless of IoU.
    separation_containment: float = 0.30


@dataclass(slots=True)
class ROI:
    x: int
    y: int
    width: int
    height: int


class FixedROI:
    """Center crop used to bound detector input and intermediate allocations."""

    def __init__(self, fraction: float = 0.82) -> None:
        self.fraction = max(0.1, min(float(fraction), 1.0))

    def bounds(self, frame: np.ndarray) -> ROI:
        height, width = frame.shape[:2]
        roi_w, roi_h = int(width * self.fraction), int(height * self.fraction)
        return ROI((width - roi_w) // 2, (height - roi_h) // 2, roi_w, roi_h)

    def crop(self, frame: np.ndarray) -> tuple[np.ndarray, ROI]:
        roi = self.bounds(frame)
        return frame[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width], roi


class EventLogger:
    """Append only meaningful state changes; never write a record per frame."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def emit(self, event: str, **details: Any) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **details,
        }
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class RateLimiter:
    """Avoid repeated alerts/events while a condition remains true."""

    def __init__(self, period_seconds: float) -> None:
        self.period_seconds = period_seconds
        self._last = 0.0

    def ready(self) -> bool:
        now = time.monotonic()
        if now - self._last < self.period_seconds:
            return False
        self._last = now
        return True


def to_display_frame(frame: np.ndarray) -> np.ndarray:
    """Return a contiguous uint8 BGR frame that Qt can wrap without copying."""

    if frame is None or frame.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(frame)


def tuning_as_dict(tuning: EdgeTuning) -> dict[str, Any]:
    """Serialisable settings shown by the dashboard and written to provenance."""

    return asdict(tuning)
