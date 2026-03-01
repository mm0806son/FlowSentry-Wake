from __future__ import annotations

from typing import Any

import numpy as np
try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from flowsentry.types import BBox, FlowRegion


def bbox_from_binary_mask(mask: Any, min_area: int = 1) -> BBox | None:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {arr.shape}")
    ys, xs = np.where(arr > 0)
    if xs.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if area < min_area:
        return None
    return (float(x1), float(y1), float(x2), float(y2))


def flow_to_mask_bboxes(
    flow: Any,
    *,
    magnitude_threshold: float,
    min_region_area: int,
) -> list[FlowRegion]:
    arr = np.asarray(flow, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError(f"Expected flow shape (H,W,2), got {arr.shape}")

    magnitude = np.linalg.norm(arr, axis=2)
    mask = (magnitude >= magnitude_threshold).astype(np.uint8)
    if mask.sum() == 0:
        return []

    regions: list[FlowRegion] = []
    if cv2 is not None:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for idx in range(1, num_labels):
            x, y, w, h, area = stats[idx]
            area = int(area)
            if area < min_region_area:
                continue
            bbox = np.array([x, y, x + w - 1, y + h - 1], dtype=np.float64)
            regions.append(FlowRegion(bbox_xyxy=bbox, area=area))
    else:
        bbox = bbox_from_binary_mask(mask, min_area=min_region_area)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            area = int(((x2 - x1) + 1) * ((y2 - y1) + 1))
            regions.append(
                FlowRegion(
                    bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float64),
                    area=area,
                )
            )
    return regions
