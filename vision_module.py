"""YOLO instance-segmentation vision engine with threaded camera ingestion.

All object recognition is delegated to one trained Ultralytics YOLO
segmentation checkpoint per experiment: ``YOLO("weights.pt")`` is called on the
camera frame and the checkpoint's own ``Results.plot()`` renders the native
instance masks back onto that frame.  There is no contour extraction, no
background subtraction, no colour thresholding and no box-tracking heuristic in
this module.

Every camera owns one background :class:`CameraWorker` thread that performs the
capture read *and* the ``results = model(frame)`` inference, so the Qt GUI thread
only ever drains already-rendered frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Callable
import logging
import math
import re
import time
import gc

import cv2
import numpy as np

from efficiency import (
    EdgeTuning,
    EventLogger,
    FixedROI,
    RateLimiter,
    to_display_frame,
)
from state_machine import (
    ActivityStateMachine,
    ExperimentFolder,
    ExperimentStep,
    load_experiment_folder,
    normalize_label,
)

logger = logging.getLogger(__name__)

#: Checkpoint expected inside a selected experiment folder.
DEFAULT_WEIGHTS_FILENAME = "weights.pt"

#: ByteTrack needs low-confidence candidates to hold an identity across frames,
#: so the tracker is fed a lower floor than the displayed confidence threshold.
TRACKER_INPUT_CONFIDENCE = 0.25

#: Number of skipped frames a cached detection survives before it is dropped.
#: Together with the persistent tracker this removes the single-frame dropouts
#: that used to make boxes blink.
HELD_FRAME_GRACE = 4


@dataclass(slots=True)
class VisionSettings:
    storage_dir: str = "storage"
    weights_path: str = ""
    #: MediaPipe Tasks assets used for the thumbs-up / open-palm gestures.
    hand_model_path: str = "hand_landmarker.task"
    face_model_path: str = "face_detector.tflite"
    confidence: float = 0.50
    iou: float = 0.50
    #: Inference resolution. 640 is the checkpoint's native size; drop to 480 to
    #: cut memory bandwidth further at the cost of mask detail.
    imgsz: int = 640
    #: Run inference on every Nth captured frame. 3 keeps the display at full
    #: camera rate while cutting GPU/CPU inference work to a third; the boxes
    #: from the last inference are re-drawn on the skipped frames.
    infer_every: int = 3
    camera_buffer_size: int = 1
    recording_fps: float = 20.0


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]
    mask_area: int = 0
    #: True when the area filter changed the class the checkpoint predicted.
    overridden: bool = False
    #: Stable identifier from the persistent tracker, when one is available.
    track_id: int | None = None
    #: Cached instance mask (uint8 0/255) so skipped frames draw the same
    #: overlay the inference frame did.
    mask: np.ndarray | None = None
    #: BGR colour, taken from the checkpoint's own class palette.
    color: tuple[int, int, int] = (0, 220, 255)

    @property
    def area(self) -> int:
        return self.box[2] * self.box[3]


def class_color(class_id: int) -> tuple[int, int, int]:
    """Colour for a class, taken from Ultralytics' own palette for consistency."""

    try:
        from ultralytics.utils.colors import Colors

        return Colors(int(class_id) % len(Colors)).color
    except Exception:
        return (0, 220, 255)


@dataclass(slots=True)
class FrameResult:
    frame: np.ndarray
    labels: list[str] = field(default_factory=list)
    instances: int = 0
    #: Capture rate of the source and rate of the inference loop, which differ
    #: whenever ``infer_every`` skips frames.
    fps: float = 0.0
    inference_fps: float = 0.0
    thumbs_up: bool = False
    forward_face: bool = False
    recording: bool = False
    recording_path: str | None = None
    #: True when this frame reused cached detections instead of running the model.
    held: bool = False
    state_event: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)


class MediaPipeTaskDetectors:
    """MediaPipe *Tasks API* wrapper; no deprecated mp.solutions code is used."""

    def __init__(self, hand_model_path: str | Path, face_model_path: str | Path) -> None:
        self.hand_landmarker = None
        self.face_detector = None
        self.error_messages: list[str] = []
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            self.mp = mp
            hand_path = Path(hand_model_path)
            if hand_path.is_file():
                hand_options = vision.HandLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=str(hand_path)),
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=1,
                    min_hand_detection_confidence=0.55,
                )
                self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
            else:
                self.error_messages.append(f"HandLandmarker task file missing: {hand_path}")
            face_path = Path(face_model_path)
            if face_path.is_file():
                face_options = vision.FaceDetectorOptions(
                    base_options=python.BaseOptions(model_asset_path=str(face_path)),
                    running_mode=vision.RunningMode.IMAGE,
                    min_detection_confidence=0.55,
                )
                self.face_detector = vision.FaceDetector.create_from_options(face_options)
            else:
                self.error_messages.append(f"FaceDetector task file missing: {face_path}")
        except Exception as exc:
            self.error_messages.append(f"MediaPipe Tasks unavailable: {exc}")

    def close(self) -> None:
        for detector in (self.hand_landmarker, self.face_detector):
            if detector is not None:
                try:
                    detector.close()
                except Exception:
                    pass

    def detect(self, bgr_frame: np.ndarray) -> tuple[bool, bool, bool]:
        """Return thumbs-up, open-palm and forward-face for the supplied frame."""

        if bgr_frame is None or bgr_frame.size == 0 or not hasattr(self, "mp"):
            return False, False, False
        image_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=image_rgb)
        thumbs_up = False
        open_palm = False
        forward_face = False
        if self.hand_landmarker is not None:
            try:
                for hand in self.hand_landmarker.detect(image).hand_landmarks:
                    thumbs_up = thumbs_up or _is_thumbs_up(hand)
                    open_palm = open_palm or _is_open_palm(hand)
            except Exception as exc:
                logger.debug("Hand task error: %s", exc)
        if self.face_detector is not None:
            try:
                faces = self.face_detector.detect(image)
                height, width = bgr_frame.shape[:2]
                for face in faces.detections:
                    box = face.bounding_box
                    center_x = (box.origin_x + box.width / 2) / max(width, 1)
                    center_y = (box.origin_y + box.height / 2) / max(height, 1)
                    fills_frame = (box.width * box.height) / max(width * height, 1) >= 0.015
                    if 0.20 <= center_x <= 0.80 and 0.15 <= center_y <= 0.85 and fills_frame:
                        forward_face = True
            except Exception as exc:
                logger.debug("Face task error: %s", exc)
        return thumbs_up, open_palm, forward_face


def _is_thumbs_up(landmarks: Any) -> bool:
    """Geometry-only thumbs-up check (works with left or right hand landmarks)."""

    if len(landmarks) < 21:
        return False
    wrist = landmarks[0]
    thumb_tip, thumb_ip, thumb_mcp = landmarks[4], landmarks[3], landmarks[2]
    finger_mcp_y = [landmarks[index].y for index in (5, 9, 13, 17)]
    thumb_extended = (
        thumb_tip.y < thumb_ip.y < thumb_mcp.y
        and thumb_tip.y < min(finger_mcp_y) - 0.04
        and math.hypot(thumb_tip.x - wrist.x, thumb_tip.y - wrist.y) > 0.16
    )
    # Non-thumb fingertips below their PIP joints in image coordinates indicate
    # closed fingers when the hand points upward.
    folded = sum(
        landmarks[tip].y > landmarks[pip].y
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
    )
    return bool(thumb_extended and folded >= 3)


def _is_open_palm(landmarks: Any) -> bool:
    if len(landmarks) < 21:
        return False
    wrist = landmarks[0]
    extended = all(
        landmarks[tip].y < landmarks[pip].y - 0.025
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
    )
    thumb_extended = math.hypot(landmarks[4].x - wrist.x, landmarks[4].y - wrist.y) > 0.16
    return bool(extended and thumb_extended)


def normalize_stream_source(raw: str) -> str:
    """Normalise typed camera input into a usable OpenCV stream URL.

    Accepts the Android *IP Webcam* convention (``192.168.1.50`` or
    ``192.168.1.50:8080/video``) as well as complete ``rtsp://``/``http://``
    URLs.  Bare addresses are expanded to the MJPEG endpoint that IP Webcam
    serves on port ``8080``.
    """

    source = (raw or "").strip()
    if not source:
        return ""
    if source.lower().startswith(("rtsp://", "http://", "https://", "rtmp://", "rtmps://", "rtp://", "udp://")):
        return source
    if source.startswith(":"):
        source = source[1:]
    if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", source):
        return f"http://{source}:8080/video"
    if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}:\d{1,5}", source):
        return f"http://{source}/video"
    return f"http://{source}"


def _apply_buffer(capture: cv2.VideoCapture, size: int) -> None:
    """Best-effort shallow frame buffer; unsupported backends silently ignore it."""

    try:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, size)
    except Exception:
        pass


def _open_capture(source: str | int) -> cv2.VideoCapture:
    """Open a local webcam (DirectShow) or a network stream (FFMPEG).

    Network streams must use FFMPEG: the Windows Media Foundation and
    DirectShow backends cannot decode MJPEG over HTTP.  Bounded open/read
    timeouts are applied so an unreachable IP address can never freeze the
    capture thread, and every backend is attempted before giving up.
    """

    if isinstance(source, int):
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        capture = cv2.VideoCapture(source, backend)
        _apply_buffer(capture, 1)
        return capture
    params: list[int] = []
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        params += [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 6000]
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        params += [cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000]
    if hasattr(cv2, "CAP_FFMPEG"):
        capture = (
            cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
            if params
            else cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        )
    else:
        capture = cv2.VideoCapture(source)
    _apply_buffer(capture, 1)
    return capture


def discover_cameras(max_index: int = 8) -> list[dict[str, str | int]]:
    """Enumerate usable local cameras; index 0 is the laptop-camera default."""

    cameras: list[dict[str, str | int]] = []
    for index in range(max_index):
        capture = _open_capture(index)
        try:
            if not capture.isOpened():
                continue
            ok, _ = capture.read()
            if ok:
                label = "Laptop/default camera" if index == 0 else f"Local camera {index}"
                cameras.append({"index": index, "label": label})
        finally:
            capture.release()
    return cameras


class YOLOSegmenter:
    """Ultralytics YOLO instance-segmentation engine.

    One checkpoint (``weights.pt``) is selected per experiment folder.  Class
    names are read straight from the checkpoint (``model.names``) so the
    vocabulary can never drift from the trained labels, and ``Results.plot()``
    draws the checkpoint's own masks, boxes and class names onto the frame.
    """

    def __init__(
        self,
        weights_path: str | Path,
        confidence: float = 0.50,
        iou: float = 0.50,
        imgsz: int = 640,
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLO

        path = Path(weights_path)
        if not path.is_file():
            raise FileNotFoundError(f"YOLO weights file not found: {path}")
        self.path = path
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = device or self._select_device()
        self.model: Any | None = YOLO(str(path))
        try:
            self.model.to(self.device)
        except Exception as exc:  # a checkpoint may not be movable to the device
            logger.debug("Model device placement ignored: %s", exc)
        self.task = str(getattr(self.model, "task", "segment") or "segment")
        names = getattr(self.model, "names", None) or {}
        self.classes: tuple[str, ...] = tuple(str(names[key]) for key in sorted(names))

    @staticmethod
    def _select_device() -> str:
        """Prefer CUDA when present, otherwise fall back to CPU inference."""

        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    @property
    def half(self) -> bool:
        return self.device.startswith("cuda")

    def warmup(self) -> None:
        """Pay the first-inference CUDA/graph cost before live frames arrive."""

        if self.model is None:
            return
        try:
            self.infer(np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8))
        except Exception as exc:
            logger.debug("YOLO warmup skipped: %s", exc)

    def describe(self) -> str:
        if self.model is None:
            return "no model loaded"
        return f"{self.path.name} [{self.task}] on {self.device}: {', '.join(self.classes) or 'no classes'}"

    def infer(self, frame: np.ndarray) -> Any:
        """Run tracked segmentation inference on one frame and return the result.

        ``model.track(frame, persist=True)`` is used instead of a bare
        ``model(frame)`` so the tracker keeps its association state between
        calls: an object that is missed on one frame is still reported (and
        keeps its track id) instead of vanishing and re-appearing, which is what
        produced the visible blinking.

        The tracker is fed a lower confidence floor than the display threshold,
        because ByteTrack needs low-confidence candidates to hold an identity;
        detections below ``self.confidence`` are filtered out afterwards, so the
        state machine still only ever sees confident objects.
        """

        if self.model is None:
            raise RuntimeError("This checkpoint was released; load an experiment first.")
        results = self.model.track(
            frame,
            persist=True,
            conf=min(self.confidence, TRACKER_INPUT_CONFIDENCE),
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if isinstance(results, (list, tuple)):
            return results[0] if results else None
        return results

    def reset_tracker(self) -> None:
        """Discard tracker association state.

        The checkpoint is shared by every camera, so the tracker must be reset
        when a (re)connects; otherwise IDs from one stream would be matched
        against another. Setting ``predictor`` to ``None`` makes Ultralytics
        build a fresh predictor, and therefore a fresh tracker, on the next call.
        """

        if self.model is None:
            return
        try:
            self.model.predictor = None
        except Exception as exc:
            logger.debug("Tracker reset skipped: %s", exc)

    def render(self, result: Any) -> np.ndarray:
        """Return the frame with the checkpoint's native masks drawn on it."""

        if result is None:
            raise ValueError("YOLO returned no result for this frame.")
        return to_display_frame(result.plot())

    @staticmethod
    def detections(result: Any, frame_shape: tuple[int, int]) -> list[Detection]:
        """Flatten a result into detection records, keeping masks and track ids.

        Instance masks are copied out of the result tensor once and cached on each
        record, so the frames that skip inference can redraw exactly the same
        overlay instead of switching to a different style of rectangle.
        """

        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            return []
        coordinates = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)
        identifiers = _track_ids(boxes)
        names = getattr(result, "names", None) or {}
        masks = _instance_masks(result, frame_shape)
        height, width = frame_shape[:2]
        found: list[Detection] = []
        for index, (coordinate, score, class_id) in enumerate(zip(coordinates, scores, class_ids)):
            name = str(names.get(int(class_id), class_id))
            left, top, right, bottom = (float(value) for value in coordinate)
            x = max(0, min(width - 1, int(left)))
            y = max(0, min(height - 1, int(top)))
            box_width = max(1, min(width - x, int(right) - x))
            box_height = max(1, min(height - y, int(bottom) - y))
            mask = masks[index] if index < len(masks) else None
            found.append(
                Detection(
                    label=name,
                    confidence=float(score),
                    box=(x, y, box_width, box_height),
                    mask_area=0 if mask is None else int((mask > 0).sum()),
                    track_id=identifiers[index] if index < len(identifiers) else None,
                    mask=mask,
                    color=class_color(int(class_id)),
                )
            )
        return found


def _track_ids(boxes: Any) -> list[int | None]:
    """Track identifiers assigned by the persistent tracker, if present."""

    identifiers = getattr(boxes, "id", None)
    if identifiers is None:
        return []
    try:
        values = identifiers.detach().cpu().numpy().tolist()
    except Exception as exc:
        logger.debug("Track ids unavailable: %s", exc)
        return []
    return [None if value is None else int(value) for value in values]


def _instance_masks(result: Any, frame_shape: tuple[int, int]) -> list[np.ndarray]:
    """Copy each instance mask out of the result as a frame-sized uint8 array."""

    masks = getattr(result, "masks", None)
    if masks is None or getattr(masks, "data", None) is None:
        return []
    try:
        data = masks.data.detach().cpu().numpy()
    except Exception as exc:
        logger.debug("Masks unavailable: %s", exc)
        return []
    height, width = frame_shape[:2]
    prepared: list[np.ndarray] = []
    for item in data:
        mask = (np.asarray(item) > 0.5).astype(np.uint8) * 255
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        prepared.append(mask)
    return prepared


def apply_area_filter(detections: list[Detection], tuning: EdgeTuning) -> list[Detection]:
    """Re-label a small dark ``blackbox`` detection as the earbud case.

    A black earbud case and a small black box are both dark plastic, so the
    trained checkpoint reports the case as ``blackbox``.  Because the two cannot
    be separated by the model, the size of the detection decides: a
    ``blackbox`` whose bounding box area is below
    ``tuning.small_object_max_area`` is re-assigned
    ``tuning.small_object_target_label``, while a large box keeps its label.

    Runs on the detection records rather than on ``result.boxes``, so the native
    masks stay untouched; the corrected name is reported through
    ``Detection.label`` and drawn on the frame by :func:`_draw_hud`.
    """

    threshold = float(tuning.small_object_max_area)
    source = normalize_label(tuning.small_object_source_label)
    target = str(tuning.small_object_target_label).strip()
    if threshold <= 0 or not source or not target:
        return detections
    for detection in detections:
        if normalize_label(detection.label) != source:
            continue
        if detection.area < threshold:
            detection.label = target
            detection.overridden = True
    return detections


def box_iou(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
    """Overlap of two ``(x, y, w, h)`` boxes, used to re-identify cached objects."""

    left = max(box1[0], box2[0])
    top = max(box1[1], box2[1])
    right = min(box1[0] + box1[2], box2[0] + box2[2])
    bottom = min(box1[1] + box1[3], box2[1] + box2[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection == 0:
        return 0.0
    union = box1[2] * box1[3] + box2[2] * box2[3] - intersection
    return intersection / union if union > 0 else 0.0


def merge_cached_detections(
    fresh: list[Detection],
    cached: list[tuple[Detection, int]],
    grace: int,
) -> tuple[list[Detection], list[tuple[Detection, int]]]:
    """Carry missed detections forward so objects stay visible between inferences.

    A single low-confidence frame must not make an object vanish: any cached
    object that is absent from ``fresh`` is kept for up to ``grace`` further
    frames. The result is what the state machine observes, so a dropout also
    cannot reset a separation streak.
    """

    if not cached or grace <= 0:
        return fresh, [(item, 0) for item in fresh]
    kept: list[tuple[Detection, int]] = []
    for detection, age in cached:
        if age >= grace:
            continue
        still_tracked = any(
            other.track_id is not None and other.track_id == detection.track_id for other in fresh
        )
        overlaps = any(
            other.label == detection.label and box_iou(other.box, detection.box) >= 0.5 for other in fresh
        )
        if not still_tracked and not overlaps:
            kept.append((detection, age + 1))
    merged = fresh + [item for item, _ in kept]
    refreshed = [(item, 0) for item in fresh] + kept
    return merged, refreshed


def boxes_by_label(detections: list[Detection]) -> dict[str, list[tuple[int, int, int, int]]]:
    """Group detection boxes by normalized class name for the spatial checks."""

    grouped: dict[str, list[tuple[int, int, int, int]]] = {}
    for detection in detections:
        grouped.setdefault(normalize_label(detection.label), []).append(detection.box)
    return grouped


def _draw_hud_placeholder() -> None:
    """Kept only so older imports fail loudly rather than silently."""


#: Every caption and outline uses the same constants, so a frame that reused
#: cached detections is pixel-for-pixel the same style as an inference frame.
BOX_THICKNESS = 2
CAPTION_FONT = cv2.FONT_HERSHEY_SIMPLEX
CAPTION_SCALE = 0.5
CAPTION_THICKNESS = 1
MASK_ALPHA = 0.45

#: Classes whose mask is drawn as an outline only. A hand held over the
#: experiment fills most of the frame with translucent colour and hides the
#: components the operator actually needs to see, so hands are not filled in.
OUTLINE_ONLY_CLASSES = frozenset({"hands", "hand", "person"})


def render_frame(
    frame: np.ndarray,
    detections: list[Detection],
    *,
    instances: int | None = None,
    fps: float = 0.0,
    recording: bool = False,
    step_text: str = "",
    held: bool = False,
) -> np.ndarray:
    """The single renderer for *every* frame, cached or freshly inferred.

    Drawing masks and outlines from the cached detection records is what removes
    the flicker: previously inference frames used Ultralytics' own ``plot()``
    while skipped frames used a different, box-only overlay, so the style
    swapped at 30 Hz. Now both paths run this function, and a held frame carries
    the same masks, colours, line thickness and captions as the inference frame
    it continues from. ``held`` only adds a small marker; nothing else changes.
    """

    height, width = frame.shape[:2]
    if detections:
        # 1. Mask fill: paint every instance into one layer, then blend once.
        # Outline-only classes are skipped so a hand cannot hide the components.
        layer = frame.copy()
        for detection in detections:
            if normalize_label(detection.label) in OUTLINE_ONLY_CLASSES:
                continue
            mask = detection.mask
            if mask is None:
                x, y, box_width, box_height = detection.box
                cv2.rectangle(
                    layer,
                    (x, y),
                    (x + box_width, y + box_height),
                    detection.color,
                    -1,
                )
                continue
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            layer[mask > 0] = detection.color
        cv2.addWeighted(layer, MASK_ALPHA, frame, 1.0 - MASK_ALPHA, 0, dst=frame)
        # 2. Outlines and captions, identical for cached and fresh detections.
        for detection in detections:
            x, y, box_width, box_height = detection.box
            cv2.rectangle(
                frame,
                (x, y),
                (x + box_width, y + box_height),
                detection.color,
                BOX_THICKNESS,
            )
            prefix = f"#{detection.track_id} " if detection.track_id is not None else ""
            suffix = " (size rule)" if detection.overridden else ""
            caption = f"{prefix}{detection.label} {detection.confidence:.2f}{suffix}"
            cv2.putText(
                frame,
                caption,
                (x + 4, max(y + 16, 62)),
                CAPTION_FONT,
                CAPTION_SCALE,
                detection.color,
                CAPTION_THICKNESS,
                cv2.LINE_AA,
            )

    # 3. Fixed HUD, drawn last so nothing overlaps the captions.
    banner = frame.copy()
    cv2.rectangle(banner, (0, 0), (width, 58), (10, 16, 26), -1)
    cv2.addWeighted(banner, 0.65, frame, 0.35, 0, dst=frame)
    total = len(detections) if instances is None else instances
    head = f"YOLO {total} instance(s) | {fps:.1f} FPS"
    if held:
        head += "  [boxes held]"
    if recording:
        head += "  REC"
    cv2.putText(frame, head, (10, 20), CAPTION_FONT, 0.52, (255, 255, 0), 1, cv2.LINE_AA)
    if step_text:
        cv2.putText(frame, step_text, (10, 44), CAPTION_FONT, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    labels = list(dict.fromkeys(detection.label for detection in detections))
    if labels:
        cv2.putText(
            frame,
            ", ".join(labels[:4]),
            (10, height - 12),
            CAPTION_FONT,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame


def release_segmenter(segmenter: YOLOSegmenter | None) -> None:
    """Drop a checkpoint from memory before another experiment is loaded.

    The model object and its CUDA tensors are dropped first, then the collector
    runs and the CUDA caching allocator is emptied, so switching experiments does
    not leave the previous ``.pt`` resident in VRAM.
    """

    if segmenter is None:
        return
    try:
        del segmenter.model
    except AttributeError:
        pass
    finally:
        segmenter.model = None  # type: ignore[assignment]
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        logger.debug("CUDA cache release skipped: %s", exc)


def _as_capture_source(source: str | int) -> str | int:
    """Resolve a user supplied source into an index, a local file or a stream URL."""

    if isinstance(source, str):
        text = source.strip()
        if text.isdigit():
            return int(text)
        if Path(text).is_file():
            return text
        return normalize_stream_source(text)
    return source


class CameraWorker:
    """Background thread owning one camera: capture read, YOLO inference, masks.

    The GUI thread never touches the camera or the model.  It only calls
    :meth:`take_result`, which returns an already annotated frame if a newer one
    has been published since the last call.
    """

    def __init__(
        self,
        camera_id: str,
        source: str | int,
        segmenter: YOLOSegmenter | None,
        settings: VisionSettings,
        tuning: EdgeTuning,
        tasks: MediaPipeTaskDetectors,
        observe: Callable[[list[str], dict[str, list[tuple[int, int, int, int]]]], dict[str, Any] | None],
        step_status: Callable[[], str],
        warnings_provider: Callable[[], list[str]],
        events: EventLogger,
        write_step_log: Callable[[Path, Path, str, str], None],
        storage_dir: Path,
    ) -> None:
        self.camera_id = camera_id
        self.source = source
        self.segmenter = segmenter
        self.settings = settings
        self.tuning = tuning
        self.tasks = tasks
        self.observe = observe
        self.step_status = step_status
        self.warnings_provider = warnings_provider
        self.events = events
        self.write_step_log = write_step_log
        self.storage_dir = storage_dir
        self.roi = FixedROI(tuning.fixed_roi_fraction)
        self.alert_limiter = RateLimiter(3.0)
        self._stop = Event()
        self._reconnect = Event()
        self._thread: Thread | None = None
        self._lock = Lock()
        self._latest: FrameResult | None = None
        self._sequence = 0
        self._taken = -1
        self.connected = False
        self.last_error = ""
        self.reconnects = 0
        self.fps = 0.0
        self.inference_fps = 0.0
        self._writer: cv2.VideoWriter | None = None
        self._recording_path: Path | None = None
        self._recording_size: tuple[int, int] | None = None
        self._start_requested = False
        self._stop_requested = False
        self._thumbs_up_frames = 0
        self._open_palm_frames = 0
        self._gesture_cooldown_until = 0.0
        self._frames = 0
        #: Most recent detections, re-drawn on the frames that skip inference.
        self._last_detections: list[Detection] = []
        #: Detections carried forward between inferences so none disappear.
        self._detection_cache: list[tuple[Detection, int]] = []
        #: How many more skipped frames the cached detections stay on screen.
        self._held = 0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, name=f"vision-{self.camera_id}", daemon=True)
        self._thread.start()

    def stop(self, wait: bool = False) -> None:
        self._stop.set()
        if wait and self._thread:
            self._thread.join(timeout=3)
        if self._thread is None or not self._thread.is_alive():
            self._thread = None
        with self._lock:
            self.connected = False

    def request_reconnect(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self.start()
            return
        self._reconnect.set()

    def reset_detections(self) -> None:
        """Forget cached boxes, e.g. when a new checkpoint is installed."""

        self._last_detections = []
        self._detection_cache = []
        self._held = 0
        self._frames = 0

    # -- GUI facing API ----------------------------------------------------
    def take_result(self) -> tuple[FrameResult | None, dict[str, Any]]:
        with self._lock:
            status = {
                "connected": self.connected,
                "error": self.last_error,
                "fps": round(self.fps, 1),
                "inference_fps": round(self.inference_fps, 1),
                "reconnects": self.reconnects,
                # True while no experiment folder has been loaded, so the UI can
                # show the awaiting prompt instead of an empty detection list.
                "awaiting_experiment": self.segmenter is None,
            }
            if self._latest is None or self._sequence == self._taken:
                return None, status
            self._taken = self._sequence
            return self._latest, status

    def request_start_recording(self) -> None:
        self._start_requested = True

    def request_stop_recording(self) -> None:
        self._stop_requested = True
        self._start_requested = False

    def close(self) -> None:
        self.stop(wait=True)
        self._release_writer("closed")

    # -- worker thread -----------------------------------------------------
    def _publish(self, result: FrameResult) -> None:
        with self._lock:
            self._latest = result
            self._sequence += 1

    def _run(self) -> None:
        delay = 0.25
        capture: cv2.VideoCapture | None = None
        prior = time.monotonic()
        prior_inference = time.monotonic()
        while not self._stop.is_set():
            try:
                if self._reconnect.is_set():
                    self._reconnect.clear()
                    capture = self._reopen(capture)
                if capture is None or not capture.isOpened():
                    capture = self._reopen(capture)
                    if capture is None:
                        self._stop.wait(delay)
                        delay = min(delay * 2, 5.0)
                        continue
                ok, frame = capture.read()
                if not ok or frame is None:
                    self.connected = False
                    self.last_error = "Frame read failed; reconnecting"
                    capture = self._reopen(capture)
                    self._stop.wait(delay)
                    delay = min(delay * 2, 5.0)
                    self.reconnects += 1
                    continue
            except Exception as exc:
                self.connected = False
                self.last_error = f"Capture error: {exc}"
                if capture is not None:
                    capture.release()
                capture = None
                self._stop.wait(delay)
                delay = min(delay * 2, 5.0)
                self.reconnects += 1
                continue
            delay = 0.25
            now = time.monotonic()
            interval = max(now - prior, 1e-3)
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / interval)
            prior = now
            try:
                self._process(frame)
                inference_interval = max(time.monotonic() - prior_inference, 1e-3)
                self.inference_fps = 0.9 * self.inference_fps + 0.1 * (1.0 / inference_interval)
                prior_inference = time.monotonic()
            except Exception as exc:
                logger.warning("Frame processing failed on %s: %s", self.camera_id, exc)
                self._publish(
                    FrameResult(
                        frame=to_display_frame(frame),
                        warnings=[f"Processing error: {exc}"],
                    )
                )
        self._release_writer("worker_stopped")

    def _reopen(self, capture: cv2.VideoCapture | None) -> cv2.VideoCapture | None:
        """Close and reopen the stream; ``None`` asks the caller to back off."""

        if capture is not None:
            capture.release()
        self.connected = False
        self.last_error = f"Connecting to {self.source}"
        capture = _open_capture(self.source)
        _apply_buffer(capture, self.settings.camera_buffer_size)
        if not capture.isOpened():
            self.last_error = f"Cannot open {self.source}"
            return None
        self.connected = True
        self.last_error = ""
        # The checkpoint is shared, so this stream needs its own tracker
        # association state; otherwise IDs would be matched across cameras.
        if self.segmenter is not None:
            self.segmenter.reset_tracker()
        self.reset_detections()
        return capture

    def _process(self, frame: np.ndarray) -> None:
        self._frames += 1
        if self.segmenter is None:
            # No experiment loaded yet: keep the preview live so the operator can
            # aim the camera while choosing a folder, and run no inference.
            self._publish(FrameResult(frame=to_display_frame(frame)))
            return
        interval = max(1, int(self.settings.infer_every))
        if self._frames % interval and self._last_detections:
            # Skipped inference. The cached masks, boxes and captions are redrawn
            # in exactly the same style as the inference frame, so nothing flickers
            # and a single dropped detection does not make an object disappear.
            self._held -= 1
            if self._held < 0:
                self._last_detections = []
            else:
                annotated = render_frame(
                    frame,
                    self._last_detections,
                    fps=self.fps,
                    recording=self._writer is not None,
                    step_text=self.step_status(),
                    held=True,
                )
                if self._writer is not None:
                    self._write_frame(annotated)
                self._publish(
                    FrameResult(
                        frame=annotated,
                        labels=list(dict.fromkeys(item.label for item in self._last_detections)),
                        instances=len(self._last_detections),
                        inference_fps=self.inference_fps,
                        fps=self.fps,
                        recording=self._writer is not None,
                        recording_path=str(self._recording_path) if self._recording_path else None,
                        held=True,
                        warnings=list(self.warnings_provider()),
                    )
                )
                return
        result = self.segmenter.infer(frame)
        fresh = self.segmenter.detections(result, frame.shape[:2])
        # Small dark objects are re-classified before anything reads the labels,
        # so the state machine, the HUD and the recording all agree.
        apply_area_filter(fresh, self.tuning)
        detections, cache = merge_cached_detections(fresh, self._detection_cache, HELD_FRAME_GRACE)
        self._detection_cache = cache
        self._last_detections = detections
        self._held = HELD_FRAME_GRACE
        labels = list(dict.fromkeys(item.label for item in detections))
        state_event = self.observe(labels, boxes_by_label(detections))
        if state_event:
            print(state_event.get("message") or state_event.get("event"), flush=True)
            self.events.emit(**{key: value for key, value in state_event.items() if key != "labels"}, labels=labels, camera=self.camera_id)
        self._handle_gestures(frame)
        annotated = render_frame(
            frame,
            detections,
            fps=self.fps,
            recording=self._writer is not None,
            step_text=self.step_status(),
            held=False,
        )
        if self._writer is not None:
            self._write_frame(annotated)
        self._publish(
            FrameResult(
                frame=annotated,
                labels=labels,
                instances=len(detections),
                inference_fps=self.inference_fps,
                fps=self.fps,
                thumbs_up=self._thumbs_up_frames > 0,
                recording=self._writer is not None,
                recording_path=str(self._recording_path) if self._recording_path else None,
                state_event=state_event,
                warnings=list(self.warnings_provider()),
            )
        )

    def _handle_gestures(self, frame: np.ndarray) -> None:
        """Thumbs-up toggles recording; an open palm stops it."""

        if self._frames % 3:
            return
        hand_frame, _ = self.roi.crop(frame)
        thumbs_up, open_palm, _ = self.tasks.detect(hand_frame)
        self._thumbs_up_frames = self._thumbs_up_frames + 1 if thumbs_up else 0
        self._open_palm_frames = self._open_palm_frames + 1 if open_palm else 0
        now = time.monotonic()
        ready = self._thumbs_up_frames >= self.tuning.gesture_frames
        stopping = (
            self._open_palm_frames >= self.tuning.gesture_frames and self._writer is not None
        )
        if self._stop_requested:
            self._stop_requested = False
            self._release_writer("manual")
        if self._start_requested and self._writer is None:
            self._start_requested = False
            self._open_writer(frame)
        if ready and now >= self._gesture_cooldown_until:
            self._gesture_cooldown_until = now + self.tuning.gesture_cooldown_seconds
            self._thumbs_up_frames = 0
            if self._writer is None:
                self._open_writer(frame)
            else:
                self._release_writer("thumbs_up_toggle")
        elif stopping and now >= self._gesture_cooldown_until:
            self._gesture_cooldown_until = now + self.tuning.gesture_cooldown_seconds
            self._open_palm_frames = 0
            self._release_writer("open_palm")

    def _open_writer(self, frame: np.ndarray) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_camera = "".join(char if char.isalnum() or char in "-_" else "_" for char in self.camera_id)
        path = self.storage_dir / f"activity_{safe_camera}_{stamp}.mp4"
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.settings.recording_fps, (width, height))
        if not writer.isOpened():
            self.events.emit("recording_error", reason="VideoWriter failed", output=str(path), camera=self.camera_id)
            return
        self._writer = writer
        self._recording_path = path
        self._recording_size = (width, height)
        self.events.emit("recording_started", video_file=path.name, camera=self.camera_id)

    def _write_frame(self, frame: np.ndarray) -> None:
        if self._writer is None:
            return
        if self._recording_size and (frame.shape[1], frame.shape[0]) != self._recording_size:
            frame = cv2.resize(frame, self._recording_size, interpolation=cv2.INTER_AREA)
        self._writer.write(np.asarray(to_display_frame(frame), dtype=np.uint8))

    def _release_writer(self, reason: str) -> None:
        if self._writer is None or self._recording_path is None:
            return
        self._writer.release()
        path = self._recording_path
        self._writer = None
        self._recording_path = None
        self._recording_size = None
        try:
            self.write_step_log(path.with_suffix(".json"), path, reason, self.camera_id)
        except Exception as exc:
            logger.warning("Could not write step log for %s: %s", path.name, exc)
        self.events.emit("recording_stopped", video_file=path.name, reason=reason, camera=self.camera_id)


class MultiCameraMonitor:
    """Shared-checkpoint coordinator: one background thread per camera.

    No checkpoint is loaded at construction time. The monitor starts in the
    *awaiting experiment* state with ``segmenter`` set to ``None``; cameras
    stream plain preview frames until :meth:`load_experiment` installs a model.
    """

    #: Shown by the UI until the user picks an experiment folder.
    AWAITING_EXPERIMENT = "Awaiting Experiment Folder Selection..."

    def __init__(
        self,
        settings: VisionSettings | None = None,
        tuning: EdgeTuning | None = None,
        steps: list[ExperimentStep] | None = None,
    ) -> None:
        self.settings = settings or VisionSettings()
        self.tuning = tuning or EdgeTuning()
        self.storage_dir = Path(self.settings.storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.events = EventLogger(self.storage_dir / "events.jsonl")
        self.tasks = MediaPipeTaskDetectors(self.settings.hand_model_path, self.settings.face_model_path)
        # Dynamic loading: no .pt is read until a folder is selected.
        self.segmenter: YOLOSegmenter | None = None
        self.experiment: ExperimentFolder | None = None
        self.state_machine = self._new_state_machine(steps or [])
        self._state_lock = Lock()
        self.workers: dict[str, CameraWorker] = {}
        self.warnings: list[str] = list(self.tasks.error_messages)
        self.closed = False

    def _new_state_machine(self, steps: list[ExperimentStep]) -> ActivityStateMachine:
        return ActivityStateMachine(
            steps,
            self.tuning.match_frames,
            self.tuning.separation_frames,
            self.tuning.separation_max_iou,
            self.tuning.separation_containment,
        )

    def _validate_area_filter(self) -> None:
        """Warn when the small-object re-label target is not a checkpoint class."""

        if self.segmenter is None:
            return
        tuning = self.tuning
        target = str(tuning.small_object_target_label).strip()
        if float(tuning.small_object_max_area) <= 0 or not target:
            return
        if not any(normalize_label(name) == normalize_label(target) for name in self.segmenter.classes):
            self.warnings.append(
                f"Area filter re-labels '{tuning.small_object_source_label}' to '{target}', "
                f"which is not a class of this checkpoint (classes: {', '.join(self.segmenter.classes)})."
            )
        elif not any(
            normalize_label(name) == normalize_label(tuning.small_object_source_label)
            for name in self.segmenter.classes
        ):
            self.warnings.append(
                f"Area filter source '{tuning.small_object_source_label}' is not a class of this "
                f"checkpoint, so no detection can be re-labelled."
            )

    # -- experiment --------------------------------------------------------
    @property
    def classes(self) -> tuple[str, ...]:
        return self.segmenter.classes if self.segmenter is not None else ()

    @property
    def model_loaded(self) -> bool:
        return self.segmenter is not None

    def load_experiment(self, folder: str | Path) -> ExperimentFolder:
        """Load a ``.pt`` checkpoint and its step file from a user picked folder.

        Blocking work: callers run this on a worker thread. The previous
        checkpoint is released first, so switching experiments never holds two
        models in VRAM, and the result is verified against the checkpoint's own
        class names before the state machine is reset.
        """

        bundle = load_experiment_folder(folder)
        previous = self.segmenter
        self.segmenter = None
        # Inference must not run against a checkpoint that is being freed.
        self._pause_workers()
        release_segmenter(previous)
        segmenter = YOLOSegmenter(
            bundle.weights_path,
            self.settings.confidence,
            self.settings.iou,
            self.settings.imgsz,
        )
        segmenter.warmup()
        # Re-parse with the checkpoint vocabulary so the required labels are
        # resolved from model.names and unresolvable ones are reported.
        bundle = load_experiment_folder(bundle.path, segmenter.classes)
        self.segmenter = segmenter
        self.experiment = bundle
        self.warnings = [message for message in self.warnings if "checkpoint" not in message]
        self._validate_area_filter()
        with self._state_lock:
            self.state_machine = self._new_state_machine(bundle.steps)
        self._resume_workers()
        self.events.emit(
            "experiment_loaded",
            folder=str(bundle.path),
            model=bundle.weights_path.name,
            steps_file=bundle.steps_path.name,
            step_count=len(bundle.steps),
            classes=list(segmenter.classes),
        )
        return bundle

    def clear_experiment(self) -> None:
        """Unload the checkpoint and return to the awaiting-experiment state."""

        previous = self.segmenter
        self.segmenter = None
        self._pause_workers()
        release_segmenter(previous)
        self.experiment = None
        with self._state_lock:
            self.state_machine = self._new_state_machine([])
        self._resume_workers()
        self.events.emit("experiment_cleared")

    def configure_experiment(self, steps: list[ExperimentStep], match_frames: int | None = None) -> None:
        with self._state_lock:
            if match_frames is None:
                self.state_machine = self._new_state_machine(steps)
            else:
                self.state_machine = ActivityStateMachine(
                    steps,
                    match_frames,
                    self.tuning.separation_frames,
                    self.tuning.separation_max_iou,
                    self.tuning.separation_containment,
                )
        self.events.emit("experiment_loaded", step_count=len(steps))

    def _observe(
        self, labels: list[str], boxes: dict[str, list[tuple[int, int, int, int]]]
    ) -> dict[str, Any] | None:
        with self._state_lock:
            return self.state_machine.observe(labels, boxes=boxes)

    def _step_status(self) -> str:
        """One-line description of the step the state machine is waiting for."""

        with self._state_lock:
            step = self.state_machine.current_step
            if step is None:
                return "Experiment complete" if self.state_machine.complete else "No experiment loaded"
            if step.requires_separation and step.separation_container:
                done, needed = self.state_machine.separation_progress
                return (
                    f"Step {step.number}/{len(self.state_machine.steps)}: {step.instruction}  "
                    f"[needs: {step.separation_target} clear of {step.separation_container} "
                    f"for {needed - done} more result(s)]"
                )
            return f"Step {step.number}/{len(self.state_machine.steps)}: {step.instruction}  [needs: {step.summary}]"

    def _pause_workers(self) -> None:
        for worker in self.workers.values():
            worker.stop()

    def _resume_workers(self) -> None:
        for worker in self.workers.values():
            worker.segmenter = self.segmenter
            worker.reset_detections()
            worker.start()

    def _write_step_log(self, json_path: Path, video_path: Path, reason: str, camera_id: str) -> None:
        with self._state_lock:
            self.state_machine.write_log(
                json_path,
                video_path,
                stop_reason=reason,
                source=str(video_path),
                camera_id=camera_id,
            )

    # -- cameras -----------------------------------------------------------
    def add_camera(self, source: str | int, camera_id: str | None = None) -> str:
        source = _as_capture_source(source)
        source_key = str(source)
        for existing_id, worker in self.workers.items():
            if str(worker.source) == source_key:
                return existing_id
        if camera_id is None:
            camera_id = f"CAM-{source}" if isinstance(source, int) else Path(source).stem or "NET"
        if camera_id in self.workers:
            suffix = 1
            while f"{camera_id}-{suffix}" in self.workers:
                suffix += 1
            camera_id = f"{camera_id}-{suffix}"
        worker = CameraWorker(
            camera_id,
            source,
            self.segmenter,
            self.settings,
            self.tuning,
            self.tasks,
            self._observe,
            self._step_status,
            lambda: self.warnings,
            self.events,
            self._write_step_log,
            self.storage_dir,
        )
        self.workers[camera_id] = worker
        worker.start()
        return camera_id

    def remove_camera(self, camera_id: str) -> None:
        worker = self.workers.pop(camera_id, None)
        if worker:
            worker.close()

    def reconnect(self, camera_id: str) -> None:
        worker = self.workers.get(camera_id)
        if worker:
            worker.request_reconnect()

    def poll(self) -> dict[str, tuple[FrameResult | None, dict[str, Any]]]:
        """Drain every worker's latest rendered frame. Never infers, never blocks."""

        updates: dict[str, tuple[FrameResult | None, dict[str, Any]]] = {}
        for camera_id, worker in self.workers.items():
            result, status = worker.take_result()
            if result is not None:
                updates[camera_id] = (result, status)
            else:
                updates[camera_id] = (None, status)
        return updates

    def start_recording(self) -> int:
        for worker in self.workers.values():
            worker.request_start_recording()
        return len(self.workers)

    def stop_recording(self) -> int:
        for worker in self.workers.values():
            worker.request_stop_recording()
        return len(self.workers)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for camera_id in list(self.workers):
            self.remove_camera(camera_id)
        self.tasks.close()
