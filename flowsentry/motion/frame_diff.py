from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from flowsentry.types import MotionEvent

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def _to_gray_u8(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3:
        if cv2 is not None:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        else:
            arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grayscale frame, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def check_frame_diff(
    prev_frame: Any,
    curr_frame: Any,
    *,
    min_area: int,
    pixel_change_ratio_threshold: float,
) -> MotionEvent:
    prev = _to_gray_u8(prev_frame)
    curr = _to_gray_u8(curr_frame)
    if prev.shape != curr.shape:
        raise ValueError(f"Frame shape mismatch: {prev.shape} vs {curr.shape}")

    if cv2 is not None:
        diff = cv2.absdiff(prev, curr)
    else:
        diff = np.abs(curr.astype(np.int16) - prev.astype(np.int16)).astype(np.uint8)

    changed = diff > 0
    changed_pixels = int(changed.sum())
    total_pixels = int(changed.size) if changed.size else 1
    change_ratio = float(changed_pixels / total_pixels)

    region_area = 0
    bbox_xyxy = None
    if changed_pixels > 0:
        if cv2 is not None:
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
                changed.astype(np.uint8), connectivity=8
            )
            for idx in range(1, num_labels):
                x, y, w, h, area = stats[idx]
                area = int(area)
                if area > region_area:
                    region_area = area
                    bbox_xyxy = np.array([x, y, x + w - 1, y + h - 1], dtype=np.float64)
        else:
            ys, xs = np.where(changed)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            region_area = int(changed_pixels)
            bbox_xyxy = np.array([x1, y1, x2, y2], dtype=np.float64)

    triggered = change_ratio >= pixel_change_ratio_threshold and region_area >= min_area
    return MotionEvent(
        triggered=bool(triggered),
        change_ratio=change_ratio,
        region_area=region_area,
        bbox_xyxy=bbox_xyxy,
    )


@dataclass
class FrameDiffMonitor:
    min_area: int = 400
    pixel_change_ratio_threshold: float = 0.01

    def update(self, _frame: Any) -> MotionEvent:
        raise NotImplementedError("FrameDiffMonitor update() is a placeholder for streaming integration.")

    def check(self, prev_frame: Any, curr_frame: Any) -> MotionEvent:
        return check_frame_diff(
            prev_frame,
            curr_frame,
            min_area=self.min_area,
            pixel_change_ratio_threshold=self.pixel_change_ratio_threshold,
        )
