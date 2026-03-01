from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from flowsentry.config import TriageConfig
from flowsentry.runtime.orchestrator import TriageOrchestrator
from flowsentry.runtime.yolo_probe import YoloProbeRecord
from flowsentry.types import FrameSignals


@dataclass(frozen=True)
class TriageReplayRecord:
    frame_index: int
    stream_id: int | None
    person_count: int
    frame_diff_triggered: bool
    flow_present: bool
    flow_consistent: bool
    flow_bbox: tuple[float, float, float, float] | None
    state: str
    yolo_enabled: bool
    alarm_triggered: bool
    alarm_reason: str
    best_iou: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "stream_id": self.stream_id,
            "person_count": self.person_count,
            "frame_diff_triggered": self.frame_diff_triggered,
            "flow_present": self.flow_present,
            "flow_consistent": self.flow_consistent,
            "flow_bbox": list(self.flow_bbox) if self.flow_bbox is not None else None,
            "state": self.state,
            "yolo_enabled": self.yolo_enabled,
            "alarm_triggered": self.alarm_triggered,
            "alarm_reason": self.alarm_reason,
            "best_iou": self.best_iou,
        }


@dataclass(frozen=True)
class TriageReplaySummary:
    frames_processed: int
    alarms_triggered: int
    first_alarm_frame_index: int | None
    alarm_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames_processed": self.frames_processed,
            "alarms_triggered": self.alarms_triggered,
            "first_alarm_frame_index": self.first_alarm_frame_index,
            "alarm_reasons": list(self.alarm_reasons),
        }


def yolo_probe_record_from_dict(obj: dict[str, Any]) -> YoloProbeRecord:
    def _bbox(v):
        return None if v is None else tuple(float(x) for x in v)

    person_bboxes = tuple(tuple(float(x) for x in box) for box in obj.get("person_bboxes", []))
    primary = _bbox(obj.get("primary_person_bbox"))
    return YoloProbeRecord(
        frame_index=int(obj["frame_index"]),
        stream_id=obj.get("stream_id"),
        person_count=int(obj["person_count"]),
        person_bboxes=person_bboxes,
        primary_person_bbox=primary,
    )


def read_yolo_probe_jsonl(path: str | Path) -> list[YoloProbeRecord]:
    in_path = Path(path)
    records: list[YoloProbeRecord] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(yolo_probe_record_from_dict(json.loads(line)))
    return records


def replay_yolo_probe_to_triage(
    records: Iterable[YoloProbeRecord],
    *,
    triage_config: TriageConfig | None = None,
) -> list[TriageReplayRecord]:
    out: list[TriageReplayRecord] = []
    orchs_by_stream: dict[int | None, TriageOrchestrator] = {}
    prev_has_person_by_stream: dict[int | None, bool] = {}

    for rec in records:
        stream_key = rec.stream_id  # `None` is treated as a single implicit stream.
        orch = orchs_by_stream.get(stream_key)
        if orch is None:
            orch = TriageOrchestrator(triage_config)
            orchs_by_stream[stream_key] = orch

        prev_has_person = prev_has_person_by_stream.get(stream_key, False)
        has_person = rec.person_count > 0
        frame_diff_triggered = has_person and not prev_has_person
        flow_present = has_person
        flow_consistent = has_person
        flow_bbox = rec.primary_person_bbox if has_person else None

        result = orch.process(
            FrameSignals(
                frame_diff_triggered=frame_diff_triggered,
                flow_present=flow_present,
                flow_consistent=flow_consistent,
                flow_bbox=flow_bbox,
                person_bboxes=rec.person_bboxes,
            )
        )
        out.append(
            TriageReplayRecord(
                frame_index=rec.frame_index,
                stream_id=rec.stream_id,
                person_count=rec.person_count,
                frame_diff_triggered=frame_diff_triggered,
                flow_present=flow_present,
                flow_consistent=flow_consistent,
                flow_bbox=flow_bbox,
                state=result.state.value,
                yolo_enabled=result.yolo_enabled,
                alarm_triggered=result.alarm.triggered,
                alarm_reason=result.alarm.reason,
                best_iou=result.match.best_iou,
            )
        )
        prev_has_person_by_stream[stream_key] = has_person

    return out


def summarize_triage_replay(records: list[TriageReplayRecord]) -> TriageReplaySummary:
    alarms = [r for r in records if r.alarm_triggered]
    first_alarm = alarms[0].frame_index if alarms else None
    reasons = tuple(sorted({r.alarm_reason for r in alarms}))
    return TriageReplaySummary(
        frames_processed=len(records),
        alarms_triggered=len(alarms),
        first_alarm_frame_index=first_alarm,
        alarm_reasons=reasons,
    )


def write_triage_replay_jsonl(records: list[TriageReplayRecord], path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    return out_path


def write_triage_replay_summary_json(summary: TriageReplaySummary, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_path
