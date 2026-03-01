from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import numpy as np

from flowsentry.config import FlowConfig
from flowsentry.motion import flow_to_mask_bboxes
from flowsentry.runtime.adapters.flow_backend_base import FlowBackendOutput
from flowsentry.types import BBox, FlowRegion


def _tensor_to_flow(tensor: Any) -> np.ndarray | None:
    """Convert tensor to flow array [H, W, 2]."""
    if tensor is None:
        return None
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()
    try:
        flow = np.asarray(tensor)
    except Exception:
        return None
    if flow.ndim == 4:
        if flow.shape[1] in (2, 4):
            flow = flow[0].transpose(1, 2, 0)[..., :2]
        elif flow.shape[-1] in (2, 4):
            flow = flow[0][..., :2]
    elif flow.ndim == 3:
        if flow.shape[0] in (2, 4):
            flow = flow.transpose(1, 2, 0)[..., :2]
        elif flow.shape[-1] in (2, 4):
            flow = flow[..., :2]
    if flow.ndim != 3 or flow.shape[-1] != 2:
        return None
    return flow


def _safe_getattr(obj: Any, attr: str) -> Any:
    try:
        return getattr(obj, attr)
    except Exception:
        return None


def _candidate_shape(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value)
    except Exception:
        img = _safe_getattr(value, "img")
        if img is None:
            return None
        try:
            arr = np.asarray(img)
        except Exception:
            return None
    if arr.shape == ():
        img = _safe_getattr(value, "img")
        if img is not None:
            try:
                arr = np.asarray(img)
            except Exception:
                pass
    return tuple(int(x) for x in arr.shape)


def _iter_flow_roots(frame_result: Any) -> list[tuple[str, Any]]:
    roots: list[tuple[str, Any]] = [("frame_result.tensor", _safe_getattr(frame_result, "tensor"))]
    for attr in ("flow_tensor", "flow", "opticalflow", "optical_flow", "prediction", "predictions"):
        val = _safe_getattr(frame_result, attr)
        if val is not None:
            roots.append((f"frame_result.{attr}", val))

    meta = _safe_getattr(frame_result, "meta")
    if meta is None:
        return roots

    if isinstance(meta, Mapping):
        preferred_keys = ("opticalflow", "flow", "tensor", "prediction", "predictions")
        seen: set[str] = set()
        for key in preferred_keys:
            if key in meta:
                seen.add(str(key))
                roots.append((f"frame_result.meta[{key!r}]", meta[key]))
        for key in meta:
            key_s = str(key)
            if key_s in seen:
                continue
            try:
                roots.append((f"frame_result.meta[{key!r}]", meta[key]))
            except Exception:
                continue
        return roots

    for attr in ("opticalflow", "flow", "tensor", "results"):
        val = _safe_getattr(meta, attr)
        if val is not None:
            roots.append((f"frame_result.meta.{attr}", val))
    return roots


def _iter_candidate_nodes(label: str, value: Any, *, max_nodes: int = 32) -> list[tuple[str, Any]]:
    nodes: list[tuple[str, Any]] = []
    queue: list[tuple[str, Any, int]] = [(label, value, 0)]
    seen: set[int] = set()
    while queue and len(nodes) < max_nodes:
        cur_label, cur_value, depth = queue.pop(0)
        if cur_value is None:
            continue
        obj_id = id(cur_value)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        nodes.append((cur_label, cur_value))
        if depth >= 2:
            continue

        for attr in (
            "tensor",
            "tensors",
            "flow",
            "opticalflow",
            "optical_flow",
            "results",
            "result",
            "data",
            "img",
        ):
            child = _safe_getattr(cur_value, attr)
            if child is None or child is cur_value:
                continue
            queue.append((f"{cur_label}.{attr}", child, depth + 1))

        if isinstance(cur_value, Mapping):
            preferred_keys = ("opticalflow", "flow", "tensor", "prediction", "predictions", "data")
            visited_keys: set[str] = set()
            for key in preferred_keys:
                if key in cur_value:
                    visited_keys.add(str(key))
                    queue.append((f"{cur_label}[{key!r}]", cur_value[key], depth + 1))
            for key in cur_value:
                key_s = str(key)
                if key_s in visited_keys:
                    continue
                try:
                    queue.append((f"{cur_label}[{key!r}]", cur_value[key], depth + 1))
                except Exception:
                    continue
        elif isinstance(cur_value, (list, tuple)):
            for idx, item in enumerate(cur_value[:8]):
                queue.append((f"{cur_label}[{idx}]", item, depth + 1))
    return nodes


def _extract_flow_from_frame_result(
    frame_result: Any,
) -> tuple[np.ndarray | None, str | None, tuple[int, ...] | None, tuple[str, ...], str | None]:
    scanned_candidates: list[str] = []
    best_flow: np.ndarray | None = None
    best_label: str | None = None
    best_raw_shape: tuple[int, ...] | None = None
    best_area = -1
    for root_label, root_value in _iter_flow_roots(frame_result):
        for cand_label, cand_value in _iter_candidate_nodes(root_label, root_value):
            scanned_candidates.append(cand_label)
            flow = _tensor_to_flow(cand_value)
            if flow is not None:
                area = int(flow.shape[0]) * int(flow.shape[1])
                if area > best_area:
                    best_flow = flow
                    best_label = cand_label
                    best_raw_shape = _candidate_shape(cand_value)
                    best_area = area
    if best_flow is not None:
        return best_flow, best_label, best_raw_shape, tuple(scanned_candidates), "tensor"
    return None, None, None, tuple(scanned_candidates), None


def _compute_bbox_iou(a: BBox | None, b: BBox | None) -> float:
    """Compute IoU between two bboxes."""
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _merge_regions_to_bbox(regions: list[FlowRegion]) -> BBox | None:
    """Select the largest connected flow region as the representative bbox."""
    if not regions:
        return None
    largest = max(regions, key=lambda r: int(r.area))
    box = largest.bbox_xyxy
    return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))


class AxeleraFlowBackend:
    """Extract optical flow from Axelera EdgeFlowNet pipeline output."""

    def __init__(
        self,
        *,
        config: FlowConfig | None = None,
        consistency_iou_threshold: float = 0.3,
        debug_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config or FlowConfig()
        self.consistency_iou_threshold = consistency_iou_threshold
        self.debug_callback = debug_callback
        self._prev_bbox: BBox | None = None

    def reset(self) -> None:
        """Reset internal state (prev frame tracking)."""
        self._prev_bbox = None

    def _compute_consistency(self, current_bbox: BBox | None) -> bool:
        """Check if current flow bbox is consistent with previous."""
        if current_bbox is None or self._prev_bbox is None:
            return False
        iou = _compute_bbox_iou(current_bbox, self._prev_bbox)
        return iou >= self.consistency_iou_threshold

    def extract(self, frame_result: Any) -> FlowBackendOutput:
        """Extract flow output from Axelera frame_result."""
        flow, source, raw_shape, scanned_candidates, decode_mode = _extract_flow_from_frame_result(frame_result)
        if self.debug_callback is not None:
            info: dict[str, Any] = {
                "flow_found": flow is not None,
                "source": source,
                "decode_mode": decode_mode,
                "raw_shape": raw_shape,
                "flow_shape": tuple(int(x) for x in flow.shape) if flow is not None else None,
                "candidate_count": len(scanned_candidates),
                "candidates": scanned_candidates[:12],
                "magnitude_p50": None,
                "magnitude_p95": None,
                "magnitude_max": None,
            }
            if flow is not None:
                magnitude = np.linalg.norm(flow.astype(np.float32), axis=2)
                info["magnitude_p50"] = round(float(np.percentile(magnitude, 50)), 6)
                info["magnitude_p95"] = round(float(np.percentile(magnitude, 95)), 6)
                info["magnitude_max"] = round(float(np.max(magnitude)), 6)
            self.debug_callback(info)

        if flow is None:
            return FlowBackendOutput(
                flow_regions=(),
                flow_bbox=None,
                flow_present=False,
                flow_consistent=False,
            )

        regions = flow_to_mask_bboxes(
            flow,
            magnitude_threshold=self.config.mask_magnitude_threshold,
            min_region_area=self.config.mask_min_region_area,
        )

        flow_bbox = _merge_regions_to_bbox(regions)
        flow_present = len(regions) > 0
        flow_consistent = self._compute_consistency(flow_bbox)

        self._prev_bbox = flow_bbox

        return FlowBackendOutput(
            flow_regions=tuple(regions),
            flow_bbox=flow_bbox,
            flow_present=flow_present,
            flow_consistent=flow_consistent,
        )

    def extract_flow_regions(self, frame_result: Any) -> tuple[FlowRegion, ...]:
        """Extract flow regions only."""
        return self.extract(frame_result).flow_regions
