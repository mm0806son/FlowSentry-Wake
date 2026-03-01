from __future__ import annotations

from flowsentry.runtime.orchestrator import TriageOrchestratorOutput


def build_debug_overlay_payload(result: TriageOrchestratorOutput) -> dict:
    return {
        "state": result.state.value,
        "optical_flow_enabled": result.optical_flow_enabled,
        "yolo_enabled": result.yolo_enabled,
        "consistency_count": result.consistency_count,
        "flow_threshold_reached": result.flow_threshold_reached,
        "alarm_triggered": result.alarm.triggered,
        "alarm_reason": result.alarm.reason,
        "best_iou": result.match.best_iou,
    }
