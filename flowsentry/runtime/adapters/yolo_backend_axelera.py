from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from flowsentry.runtime.adapters.yolo_backend_base import YoloBackendOutput
from flowsentry.types import BBox, Detection
from flowsentry.vision.yolo_filter import filter_person_detections, person_bboxes_from_detections


def _as_iterable_detections(frame_result: Any) -> list[Any]:
    detections = getattr(frame_result, "detections", None)
    if detections is not None:
        try:
            return list(detections)
        except TypeError:
            pass
        if hasattr(detections, "objects"):
            return list(detections.objects)

    meta = getattr(frame_result, "meta", None)
    if isinstance(meta, dict):
        det_meta = meta.get("detections")
        if det_meta is not None:
            if hasattr(det_meta, "objects"):
                return list(det_meta.objects)
            try:
                return list(det_meta)
            except TypeError:
                return []
    return []


def _extract_box(det: Any) -> BBox | None:
    candidates = []
    if hasattr(det, "box"):
        candidates.append(getattr(det, "box"))
    if hasattr(det, "bbox"):
        candidates.append(getattr(det, "bbox"))
    if hasattr(det, "xyxy"):
        try:
            candidates.append(det.xyxy())
        except TypeError:
            val = getattr(det, "xyxy")
            if not callable(val):
                candidates.append(val)

    for cand in candidates:
        if cand is None:
            continue
        arr = np.asarray(cand, dtype=np.float32).reshape(-1)
        if arr.size != 4:
            continue
        return tuple(float(x) for x in arr.tolist())
    return None


def _extract_score(det: Any) -> float:
    for attr in ("score", "confidence", "conf"):
        if hasattr(det, attr):
            try:
                return float(getattr(det, attr))
            except Exception:
                continue
    return 1.0


def _extract_class_id(det: Any) -> int:
    if hasattr(det, "class_id"):
        try:
            return int(getattr(det, "class_id"))
        except Exception:
            pass
    return -1


def _extract_label_name(det: Any) -> str:
    if hasattr(det, "label"):
        try:
            label = det.label
            if hasattr(label, "name"):
                return str(label.name)
            return str(label)
        except Exception:
            pass
    if hasattr(det, "class_name"):
        try:
            return str(getattr(det, "class_name"))
        except Exception:
            pass
    return ""


class AxeleraYoloBackend:
    """Extract person bounding boxes from Axelera `frame_result.detections`."""

    def __init__(
        self,
        *,
        person_labels: tuple[str, ...] = ("person",),
        person_class_ids: tuple[int, ...] = (0,),
        min_confidence: float = 0.0,
    ) -> None:
        self.person_labels = tuple(person_labels)
        self.person_class_ids = tuple(person_class_ids)
        self.min_confidence = float(min_confidence)
        self._person_label_set = {x.lower() for x in self.person_labels}

    def _is_person(self, det: Any, class_name: str, class_id: int) -> bool:
        if hasattr(det, "is_a"):
            try:
                if det.is_a("person"):
                    return True
            except Exception:
                pass
        if class_name and class_name.lower() in self._person_label_set:
            return True
        return class_id in self.person_class_ids

    def extract(self, frame_result: Any) -> YoloBackendOutput:
        all_parsed: list[Detection] = []
        person_parsed: list[Detection] = []
        for det in _as_iterable_detections(frame_result):
            bbox = _extract_box(det)
            if bbox is None:
                continue
            score = _extract_score(det)
            if score < self.min_confidence:
                continue
            class_id = _extract_class_id(det)
            class_name = _extract_label_name(det)
            is_person = self._is_person(det, class_name, class_id)
            detection = Detection(
                bbox_xyxy=bbox,
                class_id=class_id,
                class_name=class_name or f"class_{class_id}",
                confidence=score,
            )
            all_parsed.append(detection)
            if is_person:
                person_detection = Detection(
                    bbox_xyxy=bbox,
                    class_id=class_id,
                    class_name=class_name if class_name.lower() in self._person_label_set else "person",
                    confidence=score,
                )
                person_parsed.append(person_detection)

        person_dets = filter_person_detections(
            tuple(person_parsed),
            person_labels=self.person_labels,
            min_confidence=self.min_confidence,
        )
        return YoloBackendOutput(
            detections=tuple(person_parsed),
            person_bboxes=person_bboxes_from_detections(person_dets),
            all_detections=tuple(all_parsed),
        )

    def extract_person_bboxes(self, frame_result: Any) -> tuple[BBox, ...]:
        return self.extract(frame_result).person_bboxes
