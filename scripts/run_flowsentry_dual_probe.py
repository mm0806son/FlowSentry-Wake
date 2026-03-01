#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass(frozen=True)
class DualProbeRecord:
    frame_index: int
    stream_id: int | None
    flow_present: bool
    flow_consistent: bool
    flow_bbox: tuple[float, float, float, float] | None
    person_count: int
    person_bboxes: tuple[tuple[float, float, float, float], ...]
    all_detections: tuple[dict[str, Any], ...]
    state: str
    alarm_triggered: bool
    alarm_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "stream_id": self.stream_id,
            "flow_present": self.flow_present,
            "flow_consistent": self.flow_consistent,
            "flow_bbox": list(self.flow_bbox) if self.flow_bbox is not None else None,
            "person_count": self.person_count,
            "person_bboxes": [list(b) for b in self.person_bboxes],
            "all_detections": list(self.all_detections),
            "state": self.state,
            "alarm_triggered": self.alarm_triggered,
            "alarm_reason": self.alarm_reason,
        }


@dataclass
class DualProbeSummary:
    frames_processed: int
    frames_with_flow: int
    frames_with_persons: int
    frames_with_alarm: int
    total_persons: int
    alarm_triggered: bool
    first_alarm_frame: int | None
    stream_ids: tuple[int | None, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames_processed": self.frames_processed,
            "frames_with_flow": self.frames_with_flow,
            "frames_with_persons": self.frames_with_persons,
            "frames_with_alarm": self.frames_with_alarm,
            "total_persons": self.total_persons,
            "alarm_triggered": self.alarm_triggered,
            "first_alarm_frame": self.first_alarm_frame,
            "stream_ids": list(self.stream_ids),
        }


@dataclass(frozen=True)
class DualProbeAcceptanceSummary:
    preflight_ok: bool
    frames_with_joint_bboxes: int
    first_joint_bbox_frame: int | None
    has_overlay_video: bool
    has_flow_frames: bool
    has_person_frames: bool
    has_alarm: bool
    require_overlay: bool
    require_alarm: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "preflight_ok": self.preflight_ok,
            "frames_with_joint_bboxes": self.frames_with_joint_bboxes,
            "first_joint_bbox_frame": self.first_joint_bbox_frame,
            "has_overlay_video": self.has_overlay_video,
            "has_flow_frames": self.has_flow_frames,
            "has_person_frames": self.has_person_frames,
            "has_alarm": self.has_alarm,
            "require_overlay": self.require_overlay,
            "require_alarm": self.require_alarm,
            "passed": self.passed,
        }


def summarize_dual_probe_records(records: list[DualProbeRecord]) -> DualProbeSummary:
    frames_processed = len(records)
    frames_with_flow = sum(1 for r in records if r.flow_present)
    frames_with_persons = sum(1 for r in records if r.person_count > 0)
    frames_with_alarm = sum(1 for r in records if r.alarm_triggered)
    total_persons = sum(r.person_count for r in records)
    alarm_triggered = any(r.alarm_triggered for r in records)
    first_alarm = next((r.frame_index for r in records if r.alarm_triggered), None)
    stream_ids = tuple(sorted({r.stream_id for r in records if r.stream_id is not None}))
    return DualProbeSummary(
        frames_processed=frames_processed,
        frames_with_flow=frames_with_flow,
        frames_with_persons=frames_with_persons,
        frames_with_alarm=frames_with_alarm,
        total_persons=total_persons,
        alarm_triggered=alarm_triggered,
        first_alarm_frame=first_alarm,
        stream_ids=stream_ids,
    )


def write_dual_probe_jsonl(records: list[DualProbeRecord], path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    return out_path


def write_dual_probe_summary_json(summary: DualProbeSummary, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_path


def evaluate_dual_probe_acceptance(
    records: list[DualProbeRecord],
    summary: DualProbeSummary,
    *,
    preflight_ok: bool,
    overlay_video_path: str | Path | None = None,
    require_overlay: bool = False,
    require_alarm: bool = False,
) -> DualProbeAcceptanceSummary:
    frames_with_joint_bboxes = sum(
        1 for r in records if r.flow_bbox is not None and r.person_count > 0
    )
    first_joint_bbox_frame = next(
        (r.frame_index for r in records if r.flow_bbox is not None and r.person_count > 0),
        None,
    )
    has_overlay_video = True
    if require_overlay:
        has_overlay_video = overlay_video_path is not None and Path(overlay_video_path).exists()

    has_flow_frames = summary.frames_with_flow > 0
    has_person_frames = summary.frames_with_persons > 0
    has_alarm = summary.frames_with_alarm > 0

    passed = (
        preflight_ok
        and frames_with_joint_bboxes > 0
        and has_flow_frames
        and has_person_frames
        and has_overlay_video
        and (has_alarm if require_alarm else True)
    )
    return DualProbeAcceptanceSummary(
        preflight_ok=preflight_ok,
        frames_with_joint_bboxes=frames_with_joint_bboxes,
        first_joint_bbox_frame=first_joint_bbox_frame,
        has_overlay_video=has_overlay_video,
        has_flow_frames=has_flow_frames,
        has_person_frames=has_person_frames,
        has_alarm=has_alarm,
        require_overlay=require_overlay,
        require_alarm=require_alarm,
        passed=passed,
    )


def write_dual_probe_acceptance_report(
    acceptance: DualProbeAcceptanceSummary,
    path: str | Path,
    *,
    flow_network: str,
    yolo_network: str,
    source: str,
    summary: DualProbeSummary,
    jsonl_path: str | Path | None,
    summary_json_path: str | Path | None,
    overlay_video_path: str | Path | None,
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    status = "通过" if acceptance.passed else "失败"
    body = (
        "# FlowSentry 联合链路验收报告\n\n"
        f"- 结论: **{status}**\n"
        f"- flow_network: `{flow_network}`\n"
        f"- yolo_network: `{yolo_network}`\n"
        f"- source: `{source}`\n\n"
        "## 成功标志检查\n\n"
        f"1. preflight 通过: `{acceptance.preflight_ok}`\n"
        f"2. 同帧双框(frame with flow bbox + person): `{acceptance.frames_with_joint_bboxes}`"
        f" (first={acceptance.first_joint_bbox_frame})\n"
        f"3. overlay 证据可用: `{acceptance.has_overlay_video}`\n"
        f"4. 摘要指标达标(flow/person): `{acceptance.has_flow_frames}` / `{acceptance.has_person_frames}`\n"
        f"5. 报警条件({'必需' if acceptance.require_alarm else '可选'}): `{acceptance.has_alarm}`\n\n"
        "## 摘要\n\n"
        f"- frames_processed: `{summary.frames_processed}`\n"
        f"- frames_with_flow: `{summary.frames_with_flow}`\n"
        f"- frames_with_persons: `{summary.frames_with_persons}`\n"
        f"- frames_with_alarm: `{summary.frames_with_alarm}`\n\n"
        "## 产物\n\n"
        f"- jsonl: `{jsonl_path}`\n"
        f"- summary_json: `{summary_json_path}`\n"
        f"- overlay_video: `{overlay_video_path}`\n"
    )
    out_path.write_text(body, encoding="utf-8")
    return out_path


def _frame_to_ndarray(image_obj: Any, np_mod) -> Any | None:
    if image_obj is None:
        return None
    if hasattr(image_obj, "asarray"):
        frame = np_mod.asarray(image_obj.asarray())
    else:
        frame = np_mod.asarray(image_obj)
    if frame.dtype != np_mod.uint8:
        frame = frame.astype(np_mod.uint8)
    return frame


def _frame_src_timestamp_s(frame_result: Any) -> float | None:
    ts = getattr(frame_result, "src_timestamp", None)
    try:
        ts_f = float(ts)
    except (TypeError, ValueError):
        return None
    if ts_f <= 0:
        return None
    return ts_f


def _frame_age_seconds(frame_result: Any, *, now_s: float | None = None) -> float | None:
    src_s = _frame_src_timestamp_s(frame_result)
    if src_s is None:
        return None
    now = time.time() if now_s is None else now_s
    age = now - src_s
    return age if age >= 0 else 0.0


def _is_stale_frame(frame_result: Any, max_frame_age_s: float | None, *, now_s: float | None = None) -> bool:
    if max_frame_age_s is None or max_frame_age_s <= 0:
        return False
    age = _frame_age_seconds(frame_result, now_s=now_s)
    return age is not None and age > max_frame_age_s


def _pop_synced_pair(
    flow_queue,
    yolo_queue,
    *,
    tolerance_ms: int,
):
    """Pop one synced (flow_frame, yolo_frame) pair from two queues.

    If either head frame lacks src_timestamp, fallback to FIFO pairing.
    Otherwise, drop stale head frame from the earlier queue until delta <= tolerance.
    """
    tolerance_s = max(float(tolerance_ms), 0.0) / 1000.0
    while flow_queue and yolo_queue:
        flow_frame = flow_queue[0]
        yolo_frame = yolo_queue[0]
        flow_ts = _frame_src_timestamp_s(flow_frame)
        yolo_ts = _frame_src_timestamp_s(yolo_frame)

        if flow_ts is None or yolo_ts is None:
            return flow_queue.popleft(), yolo_queue.popleft()

        delta_s = flow_ts - yolo_ts
        if abs(delta_s) <= tolerance_s:
            return flow_queue.popleft(), yolo_queue.popleft()

        if delta_s < 0:
            flow_queue.popleft()
        else:
            yolo_queue.popleft()
    return None


@dataclass(frozen=True)
class DualProbeBBoxMeta:
    """Drawable flow bbox meta for SDK native display path."""

    flow_bbox: tuple[float, float, float, float]
    label: str = "Flow"
    color: tuple[int, int, int, int] = (0, 0, 255, 255)

    def draw(self, draw: Any) -> None:
        x1, y1, x2, y2 = [int(round(v)) for v in self.flow_bbox]
        draw.labelled_box((x1, y1), (x2, y2), self.label, self.color)

    def visit(self, callable_: Any, *args: Any, **kwargs: Any) -> None:
        callable_(self, *args, **kwargs)


@dataclass(frozen=True)
class DualProbeDetectionBBoxMeta:
    """Drawable YOLO bbox meta for SDK native display path."""

    bbox_xyxy: tuple[float, float, float, float]
    label: str
    color: tuple[int, int, int, int] = (0, 255, 0, 255)

    def draw(self, draw: Any) -> None:
        x1, y1, x2, y2 = [int(round(v)) for v in self.bbox_xyxy]
        draw.labelled_box((x1, y1), (x2, y2), self.label, self.color)

    def visit(self, callable_: Any, *args: Any, **kwargs: Any) -> None:
        callable_(self, *args, **kwargs)


@dataclass(frozen=True)
class DualProbeStatusMeta:
    """Drawable status banner for SDK native display path."""

    text: str
    color: tuple[int, int, int, int]
    alarm_triggered: bool = False

    def draw(self, draw: Any) -> None:
        try:
            from axelera.app.display import Font
        except Exception:
            draw.labelled_box((8, 8), (520, 40), self.text, self.color)
            return

        canvas_w, canvas_h = draw.canvas_size
        x1, y1 = 8, 8
        pad_x, pad_y = 10, 6
        base_size = max(canvas_h // 50, 12)
        font_size = int(round(base_size * 1.15)) if self.alarm_triggered else base_size
        font = Font(size=font_size, bold=self.alarm_triggered)

        max_text_w = max(0, canvas_w - x1 - 8 - pad_x * 2)
        text = self.text
        text_w, _ = draw.textsize(text, font)
        if text_w > max_text_w and max_text_w > 0:
            ellipsis = "..."
            ellipsis_w, _ = draw.textsize(ellipsis, font)
            if ellipsis_w >= max_text_w:
                text = ""
            else:
                lo, hi = 0, len(text)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    candidate = text[:mid] + ellipsis
                    candidate_w, _ = draw.textsize(candidate, font)
                    if candidate_w <= max_text_w:
                        lo = mid
                    else:
                        hi = mid - 1
                text = text[:lo] + ellipsis
            text_w, _ = draw.textsize(text, font)

        _, text_h = draw.textsize("Iy", font)
        x2 = min(canvas_w - 1, x1 + text_w + pad_x * 2)
        y2 = min(canvas_h - 1, y1 + text_h + pad_y * 2)
        text_color = (255, 255, 255, 255) if self.alarm_triggered else (20, 20, 20, 255)

        draw.rectangle((x1, y1), (x2, y2), fill=self.color, outline=None, width=1)
        draw.text((x1 + pad_x, y1 + pad_y), text, text_color, None, font)

    def visit(self, callable_: Any, *args: Any, **kwargs: Any) -> None:
        callable_(self, *args, **kwargs)


def _to_bbox_tuple(box: Any) -> tuple[float, float, float, float] | None:
    try:
        vals = [float(v) for v in box]
    except Exception:
        return None
    if len(vals) != 4:
        return None
    return (vals[0], vals[1], vals[2], vals[3])


def _select_scaled_flow_bboxes(
    flow_regions: Any,
    *,
    src_size: tuple[int, int] | None,
    dst_size: tuple[int, int] | None,
    max_boxes: int,
) -> tuple[tuple[float, float, float, float], ...]:
    if not flow_regions:
        return ()
    top_n = max(1, int(max_boxes))
    sorted_regions = sorted(
        flow_regions,
        key=lambda r: int(getattr(r, "area", 0)),
        reverse=True,
    )[:top_n]
    out: list[tuple[float, float, float, float]] = []
    for region in sorted_regions:
        raw_box = _to_bbox_tuple(getattr(region, "bbox_xyxy", None))
        if raw_box is None:
            continue
        scaled = _scale_bbox_xyxy(raw_box, src_size=src_size, dst_size=dst_size)
        if scaled is None:
            continue
        out.append(scaled)
    return tuple(out)


def _attach_flow_bboxes_overlay_meta(
    meta: Any,
    flow_bboxes: tuple[tuple[float, float, float, float], ...],
    *,
    frame_index: int,
    stream_id: int | None,
) -> Any:
    if not flow_bboxes:
        return meta

    try:
        from axelera.app.meta.base import AxMeta
    except Exception:
        return meta

    overlay_meta = meta
    if overlay_meta is None:
        overlay_meta = AxMeta(image_id=f"flowsentry-dual-probe-{stream_id}-{frame_index}")

    add_instance = getattr(overlay_meta, "add_instance", None)
    if not callable(add_instance):
        return meta

    try:
        delete_instance = getattr(overlay_meta, "delete_instance", None)
        keys = list(overlay_meta) if hasattr(overlay_meta, "__iter__") else []
        if callable(delete_instance):
            for key in keys:
                key_s = str(key)
                if key_s.startswith("flowsentry_flow_bbox_"):
                    try:
                        delete_instance(key_s)
                    except Exception:
                        pass
        for idx, bbox in enumerate(flow_bboxes):
            add_instance(
                f"flowsentry_flow_bbox_{idx}",
                DualProbeBBoxMeta(
                    flow_bbox=bbox,
                    label=f"Flow {idx + 1}",
                ),
            )
    except Exception:
        return meta
    return overlay_meta


def _is_center_in_any_flow_bbox(
    det_bbox: tuple[float, float, float, float],
    flow_bboxes: tuple[tuple[float, float, float, float], ...],
) -> bool:
    if not flow_bboxes:
        return False
    x1, y1, x2, y2 = det_bbox
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    for fx1, fy1, fx2, fy2 in flow_bboxes:
        if fx1 <= cx <= fx2 and fy1 <= cy <= fy2:
            return True
    return False


def _filter_detections_by_flow_bboxes(
    detections: tuple[Any, ...],
    flow_bboxes: tuple[tuple[float, float, float, float], ...],
) -> tuple[Any, ...]:
    if not detections or not flow_bboxes:
        return ()
    kept: list[Any] = []
    for det in detections:
        bbox = _to_bbox_tuple(getattr(det, "bbox_xyxy", None))
        if bbox is None:
            continue
        if _is_center_in_any_flow_bbox(bbox, flow_bboxes):
            kept.append(det)
    return tuple(kept)


def _attach_filtered_detection_overlay_meta(
    meta: Any,
    detections: tuple[Any, ...],
) -> Any:
    if not detections:
        return meta
    add_instance = getattr(meta, "add_instance", None)
    if not callable(add_instance):
        return meta
    for idx, det in enumerate(detections):
        bbox = _to_bbox_tuple(getattr(det, "bbox_xyxy", None))
        if bbox is None:
            continue
        class_name = str(getattr(det, "class_name", "object"))
        conf = float(getattr(det, "confidence", 0.0))
        add_instance(
            f"flowsentry_det_bbox_{idx}",
            DualProbeDetectionBBoxMeta(
                bbox_xyxy=bbox,
                label=f"{class_name} {conf:.2f}",
            ),
        )
    return meta


def _attach_status_overlay_meta(
    meta: Any,
    *,
    state: str,
    alarm_triggered: bool,
    alarm_reason: str | None,
    frame_index: int,
) -> Any:
    add_instance = getattr(meta, "add_instance", None)
    if not callable(add_instance):
        return meta
    banner = f"frame={frame_index} state={state} alarm={'ON' if alarm_triggered else 'OFF'}"
    if alarm_reason:
        banner += f" reason={alarm_reason}"
    color = (0, 0, 255, 255) if alarm_triggered else (255, 210, 0, 255)
    add_instance(
        "flowsentry_status_banner",
        DualProbeStatusMeta(
            text=banner,
            color=color,
            alarm_triggered=alarm_triggered,
        ),
    )
    return meta


def _build_filtered_display_meta(
    *,
    flow_bboxes: tuple[tuple[float, float, float, float], ...],
    detections: tuple[Any, ...],
    frame_index: int,
    stream_id: int | None,
    state: str,
    alarm_triggered: bool,
    alarm_reason: str | None,
) -> Any:
    try:
        from axelera.app.meta.base import AxMeta
    except Exception:
        return None
    meta = AxMeta(image_id=f"flowsentry-dual-probe-{stream_id}-{frame_index}")
    meta = _attach_flow_bboxes_overlay_meta(
        meta,
        flow_bboxes,
        frame_index=frame_index,
        stream_id=stream_id,
    )
    meta = _attach_filtered_detection_overlay_meta(meta, detections)
    return _attach_status_overlay_meta(
        meta,
        state=state,
        alarm_triggered=alarm_triggered,
        alarm_reason=alarm_reason,
        frame_index=frame_index,
    )


def _prefer_raw_flow_network(
    requested_network: str,
    available_candidates: tuple[str, ...] | list[str],
) -> str:
    old_aliases = {
        "edgeflownet-opticalflow",
        "ax_models/custom/edgeflownet-opticalflow.yaml",
    }
    if requested_network not in old_aliases:
        return requested_network

    preferred_order = (
        "edgeflownet-opticalflow-raw",
        "ax_models/custom/edgeflownet-opticalflow-raw.yaml",
    )
    available = set(available_candidates)
    for candidate in preferred_order:
        if candidate in available:
            return candidate

    raw_yaml = Path("ax_models/custom/edgeflownet-opticalflow-raw.yaml")
    if raw_yaml.exists():
        return str(raw_yaml)
    return requested_network


def _draw_extra_flow_bboxes_cv(
    frame: Any,
    flow_bboxes: tuple[tuple[float, float, float, float], ...],
    *,
    skip_first: bool = True,
) -> None:
    if frame is None or not flow_bboxes:
        return
    import cv2

    start_idx = 1 if skip_first else 0
    for idx, bbox in enumerate(flow_bboxes[start_idx:], start=start_idx):
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"Flow {idx + 1}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )


def _draw_filtered_detections_cv(
    frame: Any,
    detections: tuple[Any, ...],
) -> None:
    if frame is None or not detections:
        return
    import cv2

    for det in detections:
        bbox = _to_bbox_tuple(getattr(det, "bbox_xyxy", None))
        if bbox is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        class_name = str(getattr(det, "class_name", "object"))
        conf = float(getattr(det, "confidence", 0.0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{class_name} {conf:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )


def _draw_alarm_status_label_cv(
    frame: Any,
    *,
    alarm_triggered: bool,
    alarm_reason: str | None,
) -> None:
    if frame is None:
        return
    import cv2

    h, w = frame.shape[:2]
    pad_x = 12
    pad_y = 8
    text = "ALARM: ON" if alarm_triggered else "ALARM: OFF"
    if alarm_triggered and alarm_reason:
        text += f" ({alarm_reason})"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75 if alarm_triggered else 0.62
    text_thickness = 2 if alarm_triggered else 1
    text_color = (255, 255, 255) if alarm_triggered else (20, 20, 20)
    bg_color = (0, 0, 255) if alarm_triggered else (0, 220, 255)

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
    x1 = max(0, w - text_w - pad_x * 2 - 12)
    y1 = 10
    x2 = min(w - 1, x1 + text_w + pad_x * 2)
    y2 = min(h - 1, y1 + text_h + baseline + pad_y * 2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
    cv2.putText(
        frame,
        text,
        (x1 + pad_x, y1 + pad_y + text_h),
        font,
        font_scale,
        text_color,
        text_thickness,
    )


def _infer_flow_coord_size(flow_frame: Any) -> tuple[int, int] | None:
    """Infer flow coordinate space (width, height) from tensor meta."""
    from flowsentry.runtime.adapters.flow_backend_axelera import _extract_flow_from_frame_result

    flow, _, _, _, _ = _extract_flow_from_frame_result(flow_frame)
    if flow is None:
        return None
    h, w = flow.shape[:2]
    if h <= 0 or w <= 0:
        return None
    return (int(w), int(h))


def _scale_bbox_xyxy(
    bbox: tuple[float, float, float, float] | None,
    *,
    src_size: tuple[int, int] | None,
    dst_size: tuple[int, int] | None,
) -> tuple[float, float, float, float] | None:
    """Scale inclusive xyxy bbox from src size to dst size."""
    if bbox is None or src_size is None or dst_size is None:
        return bbox

    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        return bbox

    if src_w == dst_w and src_h == dst_h:
        return bbox

    sx = (dst_w - 1) / (src_w - 1) if src_w > 1 else 1.0
    sy = (dst_h - 1) / (src_h - 1) if src_h > 1 else 1.0

    x1, y1, x2, y2 = bbox
    x1_s = min(max(x1 * sx, 0.0), float(dst_w - 1))
    y1_s = min(max(y1 * sy, 0.0), float(dst_h - 1))
    x2_s = min(max(x2 * sx, 0.0), float(dst_w - 1))
    y2_s = min(max(y2 * sy, 0.0), float(dst_h - 1))
    if x2_s < x1_s:
        x1_s, x2_s = x2_s, x1_s
    if y2_s < y1_s:
        y1_s, y2_s = y2_s, y1_s
    return (x1_s, y1_s, x2_s, y2_s)


def run_dual_probe(
    *,
    flow_network: str,
    yolo_network: str,
    source: str,
    pipe_type: str = "gst",
    max_frames: int | None = None,
    triage_config: Any = None,
    stream_kwargs: dict[str, Any] | None = None,
    overlay_renderer: Any | None = None,
    video_path: str | Path | None = None,
    display: bool = False,
    display_offset: tuple[int, int] | None = None,
    stream_factory: Any | None = None,
    side_by_side: bool = False,
    use_native_display: bool = True,
    max_flow_boxes: int = 5,
    flow_magnitude_threshold: float | None = None,
    flow_min_region_area: int | None = None,
    max_frame_age_s: float | None = None,
    alarm_notifier: Any | None = None,
    ha_camera_name: str | None = None,
) -> list[DualProbeRecord]:
    import numpy as np
    from axelera.app.stream import create_inference_stream

    from flowsentry.config import FlowConfig, TriageConfig
    from flowsentry.motion import check_frame_diff
    from flowsentry.runtime.adapters import AxeleraFlowBackend, AxeleraYoloBackend
    from flowsentry.runtime.orchestrator import TriageOrchestrator

    config = triage_config or TriageConfig()
    orchestrator = TriageOrchestrator(config=config)

    flow_cfg = FlowConfig()
    if flow_magnitude_threshold is not None:
        flow_cfg.mask_magnitude_threshold = flow_magnitude_threshold
    if flow_min_region_area is not None:
        flow_cfg.mask_min_region_area = flow_min_region_area
    flow_backend = AxeleraFlowBackend(config=flow_cfg)
    yolo_backend = AxeleraYoloBackend()

    stream_kwargs = dict(stream_kwargs or {})
    if max_frame_age_s is not None and max_frame_age_s <= 0:
        max_frame_age_s = None

    if stream_factory is None:
        stream_factory = create_inference_stream

    flow_stream = stream_factory(
        network=flow_network,
        sources=[source],
        pipe_type=pipe_type,
        **stream_kwargs,
    )

    yolo_stream = stream_factory(
        network=yolo_network,
        sources=[source],
        pipe_type=pipe_type,
        **stream_kwargs,
    )

    video_writer = None
    if video_path is not None:
        video_writer = "pending"

    records: list[DualProbeRecord] = []
    prev_frame_for_diff = None
    flow_coord_size: tuple[int, int] | None = None
    # Raw flow network is stable with YOLO image buffer for display/frame-diff.
    # Legacy flow-image network keeps expected "flow background" visualization.
    prefer_yolo_image = "raw" in str(flow_network).lower()
    if display and not use_native_display:
        raise ValueError("cv2 display path has been removed. Please use --native-display for live view.")
    use_sdk_display = bool(display and use_native_display)

    def _process_pairs(native_window: Any | None = None) -> None:
        nonlocal prev_frame_for_diff, flow_coord_size, video_writer

        flow_iter = iter(flow_stream)
        yolo_iter = iter(yolo_stream)
        flow_queue = deque()
        yolo_queue = deque()
        flow_exhausted = False
        yolo_exhausted = False
        frame_index = 0

        while True:
            if max_frames is not None and frame_index >= max_frames:
                break

            if not flow_exhausted and len(flow_queue) < 2:
                try:
                    flow_queue.append(next(flow_iter))
                except StopIteration:
                    flow_exhausted = True
            if not yolo_exhausted and len(yolo_queue) < 2:
                try:
                    yolo_queue.append(next(yolo_iter))
                except StopIteration:
                    yolo_exhausted = True

            pair = _pop_synced_pair(
                flow_queue,
                yolo_queue,
                tolerance_ms=config.runtime.sync_tolerance_ms,
            )
            if pair is None:
                if (flow_exhausted or yolo_exhausted) and (not flow_queue or not yolo_queue):
                    break
                continue

            flow_frame, yolo_frame = pair
            now_s = time.time()
            if _is_stale_frame(flow_frame, max_frame_age_s, now_s=now_s):
                continue
            if _is_stale_frame(yolo_frame, max_frame_age_s, now_s=now_s):
                continue
            frame_index += 1

            if prefer_yolo_image:
                image = getattr(yolo_frame, "image", None) or getattr(flow_frame, "image", None)
            else:
                image = getattr(flow_frame, "image", None) or getattr(yolo_frame, "image", None)
            frame_for_diff = _frame_to_ndarray(image, np)

            frame_diff_triggered = False
            if prev_frame_for_diff is not None and frame_for_diff is not None:
                motion_event = check_frame_diff(
                    prev_frame_for_diff,
                    frame_for_diff,
                    min_area=config.frame_diff.min_area,
                    pixel_change_ratio_threshold=config.frame_diff.pixel_change_ratio_threshold,
                )
                frame_diff_triggered = motion_event.triggered
            if frame_for_diff is not None:
                prev_frame_for_diff = frame_for_diff

            flow_output = flow_backend.extract(flow_frame)
            yolo_output = yolo_backend.extract(yolo_frame)

            flow_bbox_mapped = flow_output.flow_bbox
            flow_bboxes_mapped: tuple[tuple[float, float, float, float], ...] = ()
            if frame_for_diff is not None:
                if flow_coord_size is None:
                    flow_coord_size = _infer_flow_coord_size(flow_frame)
                frame_h, frame_w = frame_for_diff.shape[:2]
                if flow_output.flow_bbox is not None:
                    flow_bbox_mapped = _scale_bbox_xyxy(
                        flow_output.flow_bbox,
                        src_size=flow_coord_size,
                        dst_size=(frame_w, frame_h),
                    )
                flow_bboxes_mapped = _select_scaled_flow_bboxes(
                    flow_output.flow_regions,
                    src_size=flow_coord_size,
                    dst_size=(frame_w, frame_h),
                    max_boxes=max_flow_boxes,
                )
            if not flow_bboxes_mapped and flow_bbox_mapped is not None:
                flow_bboxes_mapped = (flow_bbox_mapped,)

            from flowsentry.types import FrameSignals

            result = orchestrator.process(
                FrameSignals(
                    frame_diff_triggered=frame_diff_triggered,
                    flow_present=flow_output.flow_present,
                    flow_consistent=flow_output.flow_consistent,
                    flow_bbox=flow_bbox_mapped,
                    person_bboxes=yolo_output.person_bboxes,
                    all_detections=yolo_output.all_detections,
                )
            )
            person_bboxes = yolo_output.person_bboxes
            display_detections = _filter_detections_by_flow_bboxes(
                yolo_output.all_detections,
                flow_bboxes_mapped,
            )
            all_detections = tuple(
                {
                    "class_name": d.class_name,
                    "class_id": d.class_id,
                    "confidence": round(d.confidence, 3),
                    "bbox": list(d.bbox_xyxy),
                }
                for d in yolo_output.all_detections
            )

            if native_window is not None:
                yolo_img = getattr(yolo_frame, "image", None) or getattr(flow_frame, "image", None)
                stream_id = getattr(yolo_frame, "stream_id", getattr(flow_frame, "stream_id", None))
                display_meta = _build_filtered_display_meta(
                    flow_bboxes=flow_bboxes_mapped,
                    detections=display_detections,
                    frame_index=frame_index,
                    stream_id=stream_id,
                    state=result.state.name,
                    alarm_triggered=result.alarm.triggered,
                    alarm_reason=result.alarm.reason,
                )
                if yolo_img is not None and not native_window.is_closed:
                    native_img = yolo_img
                    if side_by_side and frame_for_diff is not None:
                        target_h, target_w = frame_for_diff.shape[:2]
                        flow_vis = _render_flow_visualization(flow_frame, (target_h, target_w))
                        if flow_vis is not None:
                            from axelera import types as ax_types

                            if flow_vis.ndim == 3 and flow_vis.shape[2] == 3:
                                flow_vis = np.ascontiguousarray(flow_vis[:, :, :3])
                            composed = np.hstack([frame_for_diff, flow_vis])
                            native_img = ax_types.Image.fromarray(
                                np.ascontiguousarray(composed),
                                ax_types.ColorFormat.RGB,
                            )
                    native_window.show(native_img, display_meta, stream_id)
                if native_window.is_closed:
                    break

            annotated_frame = None
            frame = None
            need_cv_render = overlay_renderer is not None or video_writer is not None or (display and not use_sdk_display)
            if need_cv_render:
                frame = frame_for_diff
                if frame is not None:
                    signals = FrameSignals(
                        frame_diff_triggered=frame_diff_triggered,
                        flow_present=flow_output.flow_present,
                        flow_consistent=flow_output.flow_consistent,
                        flow_bbox=flow_bbox_mapped,
                        person_bboxes=person_bboxes,
                        all_detections=yolo_output.all_detections,
                    )
                    annotated_frame = (
                        overlay_renderer.render(frame, signals, result) if overlay_renderer else frame.copy()
                    )
                    # Keep CV-rendered boxes consistent with native-display filtering.
                    _draw_filtered_detections_cv(annotated_frame, display_detections)
                    _draw_extra_flow_bboxes_cv(annotated_frame, flow_bboxes_mapped, skip_first=True)
                    _draw_alarm_status_label_cv(
                        annotated_frame,
                        alarm_triggered=result.alarm.triggered,
                        alarm_reason=result.alarm.reason,
                    )

                    if side_by_side:
                        flow_vis = _render_flow_visualization(flow_frame, (frame.shape[0], frame.shape[1]))
                        if flow_vis is not None:
                            annotated_frame = np.hstack([annotated_frame, flow_vis])

            if video_writer is not None and annotated_frame is not None:
                import cv2

                if video_writer == "pending":
                    h, w = annotated_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]
                    video_writer = cv2.VideoWriter(str(video_path), fourcc, 7.0, (w, h))
                    if not video_writer.isOpened():
                        raise RuntimeError(f"Failed to open video writer for {video_path}")
                if annotated_frame.ndim == 3 and annotated_frame.shape[2] == 3:
                    bgr = annotated_frame[:, :, ::-1].copy()
                else:
                    bgr = annotated_frame
                video_writer.write(bgr)

            records.append(
                DualProbeRecord(
                    frame_index=frame_index,
                    stream_id=getattr(flow_frame, "stream_id", None),
                    flow_present=flow_output.flow_present,
                    flow_consistent=flow_output.flow_consistent,
                    flow_bbox=flow_bbox_mapped,
                    person_count=len(person_bboxes),
                    person_bboxes=person_bboxes,
                    all_detections=all_detections,
                    state=result.state.name,
                    alarm_triggered=result.alarm.triggered,
                    alarm_reason=result.alarm.reason if result.alarm.triggered else None,
                )
            )

            if result.alarm.triggered:
                det_summary = ", ".join(
                    f"{d['class_name']}({d['confidence']:.2f})"
                    for d in all_detections
                ) or "none"
                print(
                    f"[ALARM] Frame {frame_index}: {result.alarm.reason} | detections: {det_summary}"
                )
                if alarm_notifier is not None:
                    from datetime import datetime, timezone
                    from flowsentry.runtime.alarm_notifier import AlarmEvent

                    notify_result = alarm_notifier.notify_if_needed(
                        AlarmEvent(
                            alarm_flag=True,
                            alarm_reason=result.alarm.reason or "",
                            frame_index=frame_index,
                            state=result.state.name,
                            timestamp_iso=datetime.now(timezone.utc).isoformat(),
                            stream_id=getattr(flow_frame, "stream_id", None),
                            camera=ha_camera_name or source,
                        )
                    )
                    if notify_result.sent:
                        print(
                            f"[HA] webhook sent: reason={result.alarm.reason} "
                            f"status={notify_result.status_code}"
                        )
                    elif notify_result.reason == "send_failed":
                        print(
                            f"[HA] webhook send failed: reason={result.alarm.reason} "
                            f"error={notify_result.error}"
                        )

    worker_error: Exception | None = None
    try:
        if use_sdk_display:
            from axelera.app import display as ax_display
            hardware_caps = getattr(flow_stream, "hardware_caps", None)
            opengl_cap = getattr(hardware_caps, "opengl", False)
            is_single_image = getattr(flow_stream, "is_single_image", None)
            single_image = bool(is_single_image()) if callable(is_single_image) else False

            with ax_display.App(
                renderer="auto",
                opengl=opengl_cap,
                buffering=not single_image,
            ) as app:
                wnd = app.create_window("FlowSentry Dual Probe", size=ax_display.FULL_SCREEN)

                def _worker() -> None:
                    nonlocal worker_error
                    try:
                        _process_pairs(native_window=wnd)
                    except Exception as e:  # pragma: no cover - runtime safety in display thread
                        worker_error = e

                app.start_thread(_worker, name="DualProbeThread")
                app.run(interval=1 / 10)
            if worker_error is not None:
                raise worker_error
        else:
            _process_pairs(native_window=None)

    finally:
        if video_writer is not None and video_writer != "pending":
            video_writer.release()
        for stream in (flow_stream, yolo_stream):
            stop = getattr(stream, "stop", None)
            if callable(stop):
                stop()

    return records


def _render_flow_visualization(flow_frame: Any, target_shape: tuple[int, int]):
    import numpy as np
    from flowsentry.runtime.adapters.flow_backend_axelera import (
        _extract_flow_from_frame_result,
        _safe_getattr,
    )

    flow, _, _, _, _ = _extract_flow_from_frame_result(flow_frame)
    if flow is None:
        # Fallback for legacy flow-image pipeline: show opticalflow image meta directly.
        meta = _safe_getattr(flow_frame, "meta")
        flow_img = None
        if isinstance(meta, Mapping) and "opticalflow" in meta:
            flow_img = meta["opticalflow"]
        if flow_img is not None:
            arr = None
            try:
                arr = np.asarray(flow_img)
            except Exception:
                arr = None
            if arr is None or arr.ndim != 3 or arr.shape[2] < 3:
                img = _safe_getattr(flow_img, "img")
                if img is not None:
                    try:
                        arr = np.asarray(img)
                    except Exception:
                        arr = None
            if arr is None or arr.ndim != 3 or arr.shape[2] < 3:
                return None
            flow_vis = arr[:, :, :3]
            if flow_vis.dtype != np.uint8:
                max_v = float(np.max(flow_vis)) if flow_vis.size else 0.0
                if max_v <= 1.0:
                    flow_vis = (flow_vis * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    flow_vis = flow_vis.clip(0, 255).astype(np.uint8)
            import cv2
            target_h, target_w = target_shape
            flow_vis = cv2.resize(flow_vis, (target_w, target_h))
            cv2.putText(
                flow_vis,
                "Optical Flow",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            return flow_vis
        return None

    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]

    mag = np.sqrt(fx ** 2 + fy ** 2)
    mag_norm = mag / (mag.max() + 1e-6)

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = np.arctan2(fy, fx) * 180 / np.pi / 2 + 90
    hsv[..., 1] = 255
    hsv[..., 2] = (mag_norm * 255).astype(np.uint8)

    import cv2
    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    target_h, target_w = target_shape
    flow_vis = cv2.resize(flow_vis, (target_w, target_h))

    cv2.putText(flow_vis, "Optical Flow", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return flow_vis


def main() -> None:
    from flowsentry.config import FlowConfig

    default_flow_config = FlowConfig()
    parser = argparse.ArgumentParser(description="FlowSentry dual probe (flow + YOLO fusion)")
    parser.add_argument(
        "flow_network",
        nargs="?",
        help="Flow network name, e.g. edgeflownet-opticalflow-raw",
    )
    parser.add_argument("yolo_network", nargs="?", help="YOLO network name, e.g. yolov8s-coco")
    parser.add_argument("source", nargs="?", help="Input source, e.g. fakevideo or RTSP URL")
    parser.add_argument("--pipe", dest="pipe_type", default="gst", help="Pipeline type (default: gst)")
    parser.add_argument("--frames", type=int, default=30, help="Max frames to process")
    parser.add_argument(
        "--magnitude-threshold",
        type=float,
        default=default_flow_config.mask_magnitude_threshold,
        help="Flow magnitude threshold for region detection",
    )
    parser.add_argument(
        "--min-region-area",
        type=int,
        default=default_flow_config.mask_min_region_area,
        help="Minimum region area in pixels",
    )
    parser.add_argument(
        "--max-flow-boxes",
        type=int,
        default=5,
        help="Maximum number of flow bboxes to visualize (sorted by region area)",
    )
    parser.add_argument(
        "--list-networks",
        action="store_true",
        help="List available network names and exit",
    )
    parser.add_argument(
        "--rtsp-latency",
        type=int,
        default=100,
        help="Optional RTSP latency (ms) passed to Axelera create_inference_stream",
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=7,
        help=(
            "Input frame rate for gst pipeline (default: 7). "
            "Set 0 to keep source FPS."
        ),
    )
    parser.add_argument(
        "--max-frame-age-s",
        type=float,
        default=None,
        help=(
            "Drop stale paired frames older than this age (seconds) based on src_timestamp. "
            "Unset or <=0 disables stale-frame dropping."
        ),
    )
    parser.add_argument(
        "--use-dmabuf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable DMA buffer transfers. Default enabled for lower latency; "
            "use --no-use-dmabuf if runtime stability issues are observed."
        ),
    )
    parser.add_argument("--jsonl-out", help="Write per-frame records to JSONL")
    parser.add_argument("--summary-json", help="Write summary to JSON")
    parser.add_argument(
        "--acceptance-report",
        help="Write joint acceptance report in Markdown",
    )
    parser.add_argument(
        "--acceptance-strict",
        action="store_true",
        help="Exit with non-zero code when acceptance criteria are not met",
    )
    parser.add_argument(
        "--acceptance-require-alarm",
        action="store_true",
        help="Require at least one alarm frame in acceptance result",
    )
    parser.add_argument(
        "--acceptance-require-overlay",
        action="store_true",
        help="Require overlay video evidence (typically used with --save-video)",
    )
    parser.add_argument(
        "--ha-webhook-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Home Assistant webhook notification output on alarm nodes.",
    )
    parser.add_argument(
        "--ha-webhook-url",
        default=None,
        help="Home Assistant webhook URL. If omitted, fallback to HA / HA_WEBHOOK_URL env vars.",
    )
    parser.add_argument(
        "--ha-webhook-timeout",
        type=float,
        default=2.0,
        help="Webhook timeout seconds (default: 2.0).",
    )
    parser.add_argument(
        "--ha-webhook-cooldown-seconds",
        type=float,
        default=10.0,
        help="Per-reason webhook cooldown window in seconds (default: 10).",
    )
    parser.add_argument(
        "--ha-no-object-delay-frames",
        type=int,
        default=5,
        help=(
            "For flow_no_object_detected alarms, require this many consecutive alarm frames "
            "before sending webhook (default: 5)."
        ),
    )
    parser.add_argument(
        "--ha-camera-name",
        default=None,
        help="Optional camera name included in HA payload.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate runtime env and resolve networks only",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Suppress per-frame stdout and print summary only",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Enable overlay visualization",
    )
    parser.add_argument(
        "--save-video",
        type=str,
        metavar="PATH",
        help="Save video with overlay to PATH (implies --overlay)",
    )
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="Display RGB and optical flow side by side",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show window on screen while running (e.g. on external monitor).",
    )
    parser.add_argument(
        "--native-display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use official SDK display path for live view. "
            "When enabled (default), display avoids cv2 if not using --side-by-side/--save-video."
        ),
    )
    parser.add_argument(
        "--display-offset",
        type=str,
        metavar="X,Y",
        help="Move display window to position X,Y (e.g. 1920,0 for second monitor)",
    )
    args = parser.parse_args()

    if args.use_dmabuf:
        if os.environ.get("AXELERA_USE_DMABUF") == "0":
            os.environ.pop("AXELERA_USE_DMABUF")
    else:
        os.environ["AXELERA_USE_DMABUF"] = "0"

    alarm_notifier = None
    if args.ha_webhook_enabled:
        from flowsentry.runtime.alarm_notifier import HaAlarmNotifier, HaWebhookConfig

        webhook_url = args.ha_webhook_url or os.environ.get("HA") or os.environ.get("HA_WEBHOOK_URL")
        if not webhook_url:
            raise SystemExit(
                "HA webhook is enabled but URL is missing. "
                "Use --ha-webhook-url or set HA / HA_WEBHOOK_URL."
            )
        alarm_notifier = HaAlarmNotifier(
            HaWebhookConfig(
                enabled=True,
                url=webhook_url,
                timeout_seconds=args.ha_webhook_timeout,
                cooldown_seconds=args.ha_webhook_cooldown_seconds,
                no_object_delay_frames=max(1, int(args.ha_no_object_delay_frames)),
            )
        )

    from flowsentry.runtime import (
        list_flow_network_candidates,
        list_yolo_network_candidates,
        preflight_axelera_flow_probe,
        preflight_axelera_yolo_probe,
    )

    if args.list_networks:
        print("=== Flow Networks ===")
        for name in list_flow_network_candidates():
            print(f"  {name}")
        print("\n=== YOLO Networks ===")
        for name in list_yolo_network_candidates():
            print(f"  {name}")
        return

    if not args.flow_network or not args.yolo_network:
        parser.error("flow_network and yolo_network are required")
    if not args.preflight_only and not args.source:
        parser.error("source is required unless --preflight-only is used")

    preferred_flow_network = _prefer_raw_flow_network(
        args.flow_network,
        tuple(list_flow_network_candidates()),
    )
    if preferred_flow_network != args.flow_network:
        print(
            f"[Flow] Auto-switch network: {args.flow_network} -> {preferred_flow_network} "
            "(prefer raw optical flow)"
        )

    if args.preflight_only:
        try:
            flow_preflight = preflight_axelera_flow_probe(
                network=preferred_flow_network,
                source=args.source,
                pipe_type=args.pipe_type,
            )
            yolo_preflight = preflight_axelera_yolo_probe(
                network=args.yolo_network,
                source=args.source,
                pipe_type=args.pipe_type,
            )
        except ValueError as e:
            raise SystemExit(str(e))
        print(
            f"Preflight OK:\n"
            f"  flow: {flow_preflight.resolved_network}\n"
            f"  yolo: {yolo_preflight.resolved_network}\n"
            f"  source: {args.source}"
        )
        return

    try:
        flow_preflight = preflight_axelera_flow_probe(
            network=preferred_flow_network,
            source=args.source,
            pipe_type=args.pipe_type,
        )
        yolo_preflight = preflight_axelera_yolo_probe(
            network=args.yolo_network,
            source=args.source,
            pipe_type=args.pipe_type,
        )
    except ValueError as e:
        raise SystemExit(str(e))

    stream_kwargs = {}
    if args.rtsp_latency is not None:
        stream_kwargs["rtsp_latency"] = args.rtsp_latency
    if args.frame_rate > 0:
        stream_kwargs["specified_frame_rate"] = args.frame_rate

    enable_overlay = args.overlay or args.save_video or args.side_by_side
    overlay_renderer = None
    video_path = None
    display_enabled = args.display or ((args.overlay or args.side_by_side) and not args.save_video)

    need_cv_overlay_renderer = bool(enable_overlay and (args.save_video or args.side_by_side or not args.native_display))
    if need_cv_overlay_renderer:
        from flowsentry.overlay import OverlayConfig, OverlayRenderer
        overlay_renderer = OverlayRenderer(OverlayConfig())

    if args.save_video:
        video_path = Path(args.save_video)
        video_path.parent.mkdir(parents=True, exist_ok=True)

    display_offset = None
    if args.display_offset:
        try:
            x_s, y_s = [p.strip() for p in args.display_offset.split(",", maxsplit=1)]
            display_offset = (int(x_s), int(y_s))
        except Exception as e:
            raise SystemExit(f"Invalid --display-offset '{args.display_offset}', expected X,Y") from e

    try:
        records = run_dual_probe(
            flow_network=flow_preflight.resolved_network,
            yolo_network=yolo_preflight.resolved_network,
            source=args.source,
            pipe_type=args.pipe_type,
            max_frames=args.frames,
            stream_kwargs=stream_kwargs,
            overlay_renderer=overlay_renderer,
            video_path=video_path,
            display=display_enabled,
            display_offset=display_offset,
            side_by_side=args.side_by_side,
            use_native_display=args.native_display,
            max_flow_boxes=max(1, int(args.max_flow_boxes)),
            flow_magnitude_threshold=args.magnitude_threshold,
            flow_min_region_area=args.min_region_area,
            max_frame_age_s=args.max_frame_age_s,
            alarm_notifier=alarm_notifier,
            ha_camera_name=args.ha_camera_name,
        )
    except ValueError as e:
        raise SystemExit(str(e))
    except FileNotFoundError as e:
        raise SystemExit(str(e))

    summary = summarize_dual_probe_records(records)

    if args.jsonl_out:
        write_dual_probe_jsonl(records, args.jsonl_out)
    if args.summary_json:
        write_dual_probe_summary_json(summary, args.summary_json)

    require_overlay = args.acceptance_require_overlay
    if args.save_video and (args.acceptance_report or args.acceptance_strict):
        require_overlay = True
    acceptance = evaluate_dual_probe_acceptance(
        records,
        summary,
        preflight_ok=True,
        overlay_video_path=video_path,
        require_overlay=require_overlay,
        require_alarm=args.acceptance_require_alarm,
    )
    if args.acceptance_report:
        write_dual_probe_acceptance_report(
            acceptance,
            args.acceptance_report,
            flow_network=flow_preflight.resolved_network,
            yolo_network=yolo_preflight.resolved_network,
            source=args.source,
            summary=summary,
            jsonl_path=args.jsonl_out,
            summary_json_path=args.summary_json,
            overlay_video_path=video_path,
        )

    if not args.summary_only:
        for rec in records:
            det_names = ", ".join(d["class_name"] for d in rec.all_detections) or "none"
            print(
                f"[frame={rec.frame_index}] "
                f"flow={rec.flow_present} consistent={rec.flow_consistent} "
                f"detections=[{det_names}] state={rec.state} "
                f"alarm={rec.alarm_triggered}"
            )

    print(
        f"Summary: frames={summary.frames_processed}, "
        f"with_flow={summary.frames_with_flow}, "
        f"with_persons={summary.frames_with_persons}, "
        f"alarms={summary.frames_with_alarm}, "
        f"alarm_triggered={summary.alarm_triggered}"
    )
    print(
        f"Acceptance: passed={acceptance.passed}, "
        f"joint_bbox_frames={acceptance.frames_with_joint_bboxes}, "
        f"first_joint_frame={acceptance.first_joint_bbox_frame}, "
        f"overlay_ok={acceptance.has_overlay_video}, "
        f"alarm_ok={acceptance.has_alarm}"
    )
    if args.acceptance_strict and not acceptance.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
