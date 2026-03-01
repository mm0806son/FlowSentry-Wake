from __future__ import annotations

import json

from flowsentry.config import TriageConfig
from flowsentry.runtime.triage_replay import (
    TriageReplayRecord,
    read_yolo_probe_jsonl,
    replay_yolo_probe_to_triage,
    summarize_triage_replay,
    write_triage_replay_jsonl,
    write_triage_replay_summary_json,
    yolo_probe_record_from_dict,
)
from flowsentry.runtime.yolo_probe import YoloProbeRecord


def test_yolo_probe_record_from_dict_parses_bbox_lists():
    obj = {
        "frame_index": 7,
        "stream_id": 1,
        "person_count": 2,
        "person_bboxes": [[1, 2, 3, 4], [5.5, 6.5, 7.5, 8.5]],
        "primary_person_bbox": [5.5, 6.5, 7.5, 8.5],
    }

    rec = yolo_probe_record_from_dict(obj)

    assert rec.frame_index == 7
    assert rec.stream_id == 1
    assert rec.person_bboxes == ((1.0, 2.0, 3.0, 4.0), (5.5, 6.5, 7.5, 8.5))
    assert rec.primary_person_bbox == (5.5, 6.5, 7.5, 8.5)


def test_read_probe_jsonl_and_replay_to_alarm(tmp_path):
    probe_records = [
        YoloProbeRecord(1, 0, 0, (), None),
        YoloProbeRecord(2, 0, 1, ((10.0, 10.0, 50.0, 50.0),), (10.0, 10.0, 50.0, 50.0)),
        YoloProbeRecord(3, 0, 1, ((10.0, 10.0, 50.0, 50.0),), (10.0, 10.0, 50.0, 50.0)),
    ]
    probe_jsonl = tmp_path / "probe.jsonl"
    probe_jsonl.write_text(
        "\n".join(json.dumps(r.to_dict()) for r in probe_records) + "\n",
        encoding="utf-8",
    )

    loaded = read_yolo_probe_jsonl(probe_jsonl)

    cfg = TriageConfig()
    cfg.flow.consistency_frames_threshold = 2
    replay = replay_yolo_probe_to_triage(loaded, triage_config=cfg)

    assert [r.state for r in replay] == ["standby", "flow_active", "alarm"]
    assert replay[-1].alarm_triggered is True
    assert replay[-1].alarm_reason == "person_iou_match"
    assert replay[-1].best_iou is not None
    assert replay[-1].best_iou >= cfg.fusion.iou_threshold


def test_triage_replay_summary_and_outputs(tmp_path):
    records = [
        TriageReplayRecord(1, 0, 0, False, False, False, None, "standby", False, False, "not_in_yolo_stage", None),
        TriageReplayRecord(2, 0, 1, True, True, True, (0, 0, 10, 10), "alarm", True, True, "person_iou_match", 1.0),
        TriageReplayRecord(3, 0, 1, False, True, True, (0, 0, 10, 10), "alarm", True, True, "person_iou_match", 1.0),
    ]
    summary = summarize_triage_replay(records)

    assert summary.frames_processed == 3
    assert summary.alarms_triggered == 2
    assert summary.first_alarm_frame_index == 2
    assert summary.alarm_reasons == ("person_iou_match",)

    jsonl_path = write_triage_replay_jsonl(records, tmp_path / "triage" / "events.jsonl")
    summary_path = write_triage_replay_summary_json(summary, tmp_path / "triage" / "summary.json")

    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    obj0 = json.loads(lines[0])
    assert obj0["state"] == "standby"
    assert obj0["alarm_triggered"] is False

    summary_obj = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_obj["alarms_triggered"] == 2
    assert summary_obj["alarm_reasons"] == ["person_iou_match"]


def test_triage_replay_isolates_state_for_interleaved_streams():
    records = [
        YoloProbeRecord(1, 0, 0, (), None),
        YoloProbeRecord(2, 0, 1, ((10.0, 10.0, 50.0, 50.0),), (10.0, 10.0, 50.0, 50.0)),
        YoloProbeRecord(3, 1, 1, ((20.0, 20.0, 60.0, 60.0),), (20.0, 20.0, 60.0, 60.0)),
        YoloProbeRecord(4, 1, 1, ((20.0, 20.0, 60.0, 60.0),), (20.0, 20.0, 60.0, 60.0)),
    ]

    cfg = TriageConfig()
    cfg.flow.consistency_frames_threshold = 2
    replay = replay_yolo_probe_to_triage(records, triage_config=cfg)

    assert [r.state for r in replay] == ["standby", "flow_active", "flow_active", "alarm"]
    assert replay[2].stream_id == 1
    assert replay[2].frame_diff_triggered is True
    assert replay[2].alarm_triggered is False
    assert replay[3].stream_id == 1
    assert replay[3].alarm_triggered is True
    assert replay[3].alarm_reason == "person_iou_match"
