from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from flowsentry.overlay.config import OverlayConfig

if TYPE_CHECKING:
    from flowsentry.runtime.orchestrator import TriageOrchestratorOutput
    from flowsentry.types import FrameSignals


class OverlayRenderer:
    def __init__(self, config: OverlayConfig | None = None):
        self.config = config or OverlayConfig()
    
    def render(
        self,
        frame: np.ndarray,
        signals: "FrameSignals",
        result: "TriageOrchestratorOutput",
    ) -> np.ndarray:
        annotated = frame.copy()
        
        self._draw_flow_bbox(annotated, signals)
        self._draw_person_bboxes(annotated, signals)
        self._draw_status(annotated, signals, result)
        
        if result.alarm.triggered:
            self._draw_alarm_border(annotated)
        
        return annotated
    
    def _draw_flow_bbox(self, frame: np.ndarray, signals: "FrameSignals") -> None:
        if signals.flow_bbox is None:
            return
        
        x1, y1, x2, y2 = [int(v) for v in signals.flow_bbox]
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            self.config.flow_bbox_color,
            self.config.line_thickness,
        )
        cv2.putText(
            frame,
            "Flow",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale * 0.8,
            self.config.flow_bbox_color,
            self.config.font_thickness,
        )
    
    def _draw_person_bboxes(self, frame: np.ndarray, signals: "FrameSignals") -> None:
        for i, bbox in enumerate(signals.person_bboxes):
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                self.config.person_bbox_color,
                self.config.line_thickness,
            )
            label = f"Person {i + 1}" if len(signals.person_bboxes) > 1 else "Person"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale * 0.8,
                self.config.person_bbox_color,
                self.config.font_thickness,
            )
    
    def _draw_status(
        self,
        frame: np.ndarray,
        signals: "FrameSignals",
        result: "TriageOrchestratorOutput",
    ) -> None:
        x, y = self.config.state_text_offset
        
        state_text = f"State: {result.state}"
        cv2.putText(
            frame,
            state_text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            self.config.state_color,
            self.config.font_thickness,
        )
        
        x2, y2 = self.config.info_text_offset
        consistency_text = f"Consistency: {result.consistency_count}"
        cv2.putText(
            frame,
            consistency_text,
            (x2, y2),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale * 0.8,
            self.config.state_color,
            self.config.font_thickness,
        )
        
        if result.alarm.triggered:
            y3 = y2 + 25
            alarm_text = f"ALARM: {result.alarm.reason}"
            cv2.putText(
                frame,
                alarm_text,
                (x2, y3),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale * 0.8,
                self.config.alarm_border_color,
                self.config.font_thickness,
            )
    
    def _draw_alarm_border(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        t = self.config.alarm_border_thickness
        color = self.config.alarm_border_color
        
        cv2.rectangle(frame, (0, 0), (w, t), color, -1)
        cv2.rectangle(frame, (0, h - t), (w, h), color, -1)
        cv2.rectangle(frame, (0, 0), (t, h), color, -1)
        cv2.rectangle(frame, (w - t, 0), (w, h), color, -1)
