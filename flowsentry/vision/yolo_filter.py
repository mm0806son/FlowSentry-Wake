from __future__ import annotations

from flowsentry.types import BBox, Detection


def filter_person_detections(
    detections: tuple[Detection, ...],
    *,
    person_labels: tuple[str, ...] = ("person",),
    min_confidence: float = 0.0,
) -> tuple[Detection, ...]:
    labels = {x.lower() for x in person_labels}
    out: list[Detection] = []
    for det in detections:
        class_name = (det.class_name or "").lower()
        if class_name not in labels:
            continue
        if det.confidence < min_confidence:
            continue
        out.append(det)
    out.sort(key=lambda d: d.confidence, reverse=True)
    return tuple(out)


def person_bboxes_from_detections(detections: tuple[Detection, ...]) -> tuple[BBox, ...]:
    return tuple(tuple(float(v) for v in det.bbox_xyxy) for det in detections)


def select_primary_person_bbox(person_bboxes: tuple[BBox, ...]) -> BBox | None:
    if not person_bboxes:
        return None
    return max(person_bboxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
