from __future__ import annotations

from flowsentry.types import AlarmDecision, BBox, Detection


class AlarmPolicy:
    def __init__(self, iou_threshold: float) -> None:
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        *,
        flow_present: bool,
        flow_bbox: BBox | None,
        person_bboxes: tuple[BBox, ...],
        best_iou: float | None,
        best_person_bbox: BBox | None,
        all_detections: tuple[Detection, ...] = (),
    ) -> AlarmDecision:
        if not flow_present:
            return AlarmDecision(
                False, reason="no_flow", flow_bbox=flow_bbox, all_detections=all_detections
            )

        if not person_bboxes:
            if all_detections:
                return AlarmDecision(
                    False,
                    reason="non_person_object_detected",
                    best_iou=best_iou,
                    matched=False,
                    flow_bbox=flow_bbox,
                    all_detections=all_detections,
                )
            return AlarmDecision(
                True,
                reason="flow_no_object_detected",
                best_iou=best_iou,
                matched=False,
                flow_bbox=flow_bbox,
                all_detections=all_detections,
            )

        matched = best_iou is not None and best_iou >= self.iou_threshold
        reason = "person_iou_match" if matched else "person_iou_below_threshold"
        return AlarmDecision(
            True,
            reason=reason,
            best_iou=best_iou,
            matched=matched,
            flow_bbox=flow_bbox,
            person_bbox=best_person_bbox,
            all_detections=all_detections,
        )
