from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flowsentry.types import BBox

try:
    from axelera.app.model_utils.box import box_iou_1_to_many as _sdk_box_iou_1_to_many
except Exception:  # pragma: no cover - fallback for lightweight environments
    _sdk_box_iou_1_to_many = None


def _fallback_box_iou_1_to_many(box1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    x_a = np.maximum(x11, x21.T)
    y_a = np.maximum(y11, y21.T)
    x_b = np.minimum(x12, x22.T)
    y_b = np.minimum(y12, y22.T)
    inter = np.maximum(x_b - x_a + 1, 0) * np.maximum(y_b - y_a + 1, 0)
    area1 = (x12 - x11 + 1) * (y12 - y11 + 1)
    area2 = (x22 - x21 + 1) * (y22 - y21 + 1)
    return inter / (area1 + area2.T - inter)


def _box_iou_1_to_many(box1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    if _sdk_box_iou_1_to_many is not None:
        return _sdk_box_iou_1_to_many(box1, bboxes2)
    return _fallback_box_iou_1_to_many(box1, bboxes2)


@dataclass(frozen=True)
class IoUMatchResult:
    matched: bool
    best_iou: float | None
    best_person_bbox: BBox | None
    threshold: float


class IoUMatcher:
    def __init__(self, iou_threshold: float) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        self.iou_threshold = iou_threshold

    def match(self, flow_bbox: BBox | None, person_bboxes: tuple[BBox, ...]) -> IoUMatchResult:
        if flow_bbox is None or not person_bboxes:
            return IoUMatchResult(
                matched=False,
                best_iou=None,
                best_person_bbox=None,
                threshold=self.iou_threshold,
            )

        flow_np = np.asarray(flow_bbox, dtype=np.float32)
        person_np = np.asarray(person_bboxes, dtype=np.float32)
        if person_np.ndim != 2 or person_np.shape[1] != 4:
            raise ValueError(f"Expected person_bboxes shape (N,4), got {person_np.shape}")

        ious = np.asarray(_box_iou_1_to_many(flow_np, person_np), dtype=np.float32).reshape(-1)
        if ious.size == 0:
            return IoUMatchResult(False, None, None, self.iou_threshold)

        best_idx = int(np.argmax(ious))
        best_iou = float(ious[best_idx])
        best_person_bbox = tuple(float(x) for x in person_np[best_idx].tolist())
        return IoUMatchResult(
            matched=best_iou >= self.iou_threshold,
            best_iou=best_iou,
            best_person_bbox=best_person_bbox,
            threshold=self.iou_threshold,
        )
