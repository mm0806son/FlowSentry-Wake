from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flowsentry.config import TriageConfig
from flowsentry.fsm import TriageState, TriageStateMachine
from flowsentry.fusion import AlarmPolicy, IoUMatchResult, IoUMatcher
from flowsentry.motion import MotionConsistencyCounter
from flowsentry.runtime.adapters.flow_backend_base import FlowBackend
from flowsentry.runtime.adapters.yolo_backend_base import YoloBackend
from flowsentry.types import AlarmDecision, BBox, Detection, FrameSignals


@dataclass(frozen=True)
class TriageOrchestratorOutput:
    state: TriageState
    optical_flow_enabled: bool
    yolo_enabled: bool
    consistency_count: int
    flow_threshold_reached: bool
    match: IoUMatchResult
    alarm: AlarmDecision


class TriageOrchestrator:
    def __init__(self, config: TriageConfig | None = None) -> None:
        self.config = config or TriageConfig()
        self.consistency = MotionConsistencyCounter(self.config.flow.consistency_frames_threshold)
        self.matcher = IoUMatcher(self.config.fusion.iou_threshold)
        self.policy = AlarmPolicy(self.config.fusion.iou_threshold)
        self.fsm = TriageStateMachine(self.config.runtime.no_motion_reset_frames)

    def reset(self) -> None:
        self.consistency.reset()
        self.fsm.reset()

    def _should_count_consistency(self, signals: FrameSignals) -> bool:
        return self.fsm.state != TriageState.STANDBY or signals.frame_diff_triggered

    @staticmethod
    def _is_bbox_center_in_flow_region(bbox: BBox, flow_bbox: BBox | None) -> bool:
        if flow_bbox is None:
            return False
        x1, y1, x2, y2 = bbox
        cx = (float(x1) + float(x2)) * 0.5
        cy = (float(y1) + float(y2)) * 0.5
        fx1, fy1, fx2, fy2 = flow_bbox
        return float(fx1) <= cx <= float(fx2) and float(fy1) <= cy <= float(fy2)

    def _filter_person_bboxes_in_flow_region(
        self,
        person_bboxes: tuple[BBox, ...],
        flow_bbox: BBox | None,
    ) -> tuple[BBox, ...]:
        if not person_bboxes or flow_bbox is None:
            return ()
        return tuple(b for b in person_bboxes if self._is_bbox_center_in_flow_region(b, flow_bbox))

    def _filter_detections_in_flow_region(
        self,
        detections: tuple[Detection, ...],
        flow_bbox: BBox | None,
    ) -> tuple[Detection, ...]:
        if not detections or flow_bbox is None:
            return ()
        kept: list[Detection] = []
        for det in detections:
            try:
                bbox = tuple(float(v) for v in det.bbox_xyxy)
            except Exception:
                continue
            if len(bbox) != 4:
                continue
            if self._is_bbox_center_in_flow_region(bbox, flow_bbox):
                kept.append(det)
        return tuple(kept)

    def process(self, signals: FrameSignals) -> TriageOrchestratorOutput:
        if self._should_count_consistency(signals):
            flow_threshold_reached = self.consistency.update(signals.flow_consistent)
        else:
            self.consistency.reset()
            flow_threshold_reached = False

        yolo_stage_now_or_next = (
            self.fsm.state in {TriageState.YOLO_VERIFY, TriageState.ALARM}
            or (self.fsm.state == TriageState.FLOW_ACTIVE and flow_threshold_reached)
        )

        if yolo_stage_now_or_next:
            person_bboxes_in_flow = self._filter_person_bboxes_in_flow_region(
                signals.person_bboxes, signals.flow_bbox
            )
            detections_in_flow = self._filter_detections_in_flow_region(
                signals.all_detections, signals.flow_bbox
            )
            match = self.matcher.match(signals.flow_bbox, person_bboxes_in_flow)
            alarm = self.policy.evaluate(
                flow_present=signals.flow_present,
                flow_bbox=signals.flow_bbox,
                person_bboxes=person_bboxes_in_flow,
                best_iou=match.best_iou,
                best_person_bbox=match.best_person_bbox,
                all_detections=detections_in_flow,
            )
        else:
            match = IoUMatchResult(
                matched=False,
                best_iou=None,
                best_person_bbox=None,
                threshold=self.config.fusion.iou_threshold,
            )
            alarm = AlarmDecision(False, reason="not_in_yolo_stage")

        step = self.fsm.step(
            frame_diff_triggered=signals.frame_diff_triggered,
            flow_present=signals.flow_present,
            flow_threshold_reached=flow_threshold_reached,
            alarm_triggered=alarm.triggered,
        )

        if step.state == TriageState.STANDBY:
            self.consistency.reset()

        return TriageOrchestratorOutput(
            state=step.state,
            optical_flow_enabled=step.optical_flow_enabled,
            yolo_enabled=step.yolo_enabled,
            consistency_count=self.consistency.count,
            flow_threshold_reached=flow_threshold_reached,
            match=match,
            alarm=alarm,
        )

    def process_with_yolo_backend(
        self,
        *,
        frame_diff_triggered: bool,
        flow_present: bool,
        flow_consistent: bool,
        flow_bbox,
        yolo_backend: YoloBackend,
        yolo_frame_result: Any,
        timestamp_ms: int | None = None,
    ) -> TriageOrchestratorOutput:
        yolo_output = yolo_backend.extract(yolo_frame_result)
        return self.process(
            FrameSignals(
                frame_diff_triggered=frame_diff_triggered,
                flow_present=flow_present,
                flow_consistent=flow_consistent,
                flow_bbox=flow_bbox,
                person_bboxes=yolo_output.person_bboxes,
                all_detections=yolo_output.all_detections,
                timestamp_ms=timestamp_ms,
            )
        )

    def process_with_flow_backend(
        self,
        *,
        frame_diff_triggered: bool,
        flow_backend: FlowBackend,
        flow_frame_result: Any,
        timestamp_ms: int | None = None,
    ) -> TriageOrchestratorOutput:
        flow_output = flow_backend.extract(flow_frame_result)
        return self.process(
            FrameSignals(
                frame_diff_triggered=frame_diff_triggered,
                flow_present=flow_output.flow_present,
                flow_consistent=flow_output.flow_consistent,
                flow_bbox=flow_output.flow_bbox,
                person_bboxes=(),
                timestamp_ms=timestamp_ms,
            )
        )

    def process_with_both_backends(
        self,
        *,
        frame_diff_triggered: bool,
        flow_backend: FlowBackend,
        flow_frame_result: Any,
        yolo_backend: YoloBackend,
        yolo_frame_result: Any,
        timestamp_ms: int | None = None,
    ) -> TriageOrchestratorOutput:
        flow_output = flow_backend.extract(flow_frame_result)
        yolo_output = yolo_backend.extract(yolo_frame_result)
        return self.process(
            FrameSignals(
                frame_diff_triggered=frame_diff_triggered,
                flow_present=flow_output.flow_present,
                flow_consistent=flow_output.flow_consistent,
                flow_bbox=flow_output.flow_bbox,
                person_bboxes=yolo_output.person_bboxes,
                all_detections=yolo_output.all_detections,
                timestamp_ms=timestamp_ms,
            )
        )
