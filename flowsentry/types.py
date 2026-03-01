from __future__ import annotations

from dataclasses import dataclass
from typing import Any


BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    bbox_xyxy: Any
    class_id: int
    class_name: str
    confidence: float


@dataclass(frozen=True)
class FlowRegion:
    bbox_xyxy: Any
    area: int


@dataclass(frozen=True)
class MotionEvent:
    triggered: bool
    change_ratio: float
    region_area: int
    bbox_xyxy: Any | None = None


@dataclass(frozen=True)
class AlarmDecision:
    triggered: bool
    reason: str
    best_iou: float | None = None
    matched: bool | None = None
    flow_bbox: BBox | None = None
    person_bbox: BBox | None = None
    all_detections: tuple[Detection, ...] = ()


@dataclass(frozen=True)
class FrameSignals:
    frame_diff_triggered: bool
    flow_present: bool
    flow_consistent: bool
    flow_bbox: BBox | None = None
    person_bboxes: tuple[BBox, ...] = ()
    all_detections: tuple[Detection, ...] = ()
    timestamp_ms: int | None = None
