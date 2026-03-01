from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


class TestDualProbeRecordAndSummary:
    def test_dual_probe_record_to_dict(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        
        from run_flowsentry_dual_probe import DualProbeRecord

        record = DualProbeRecord(
            frame_index=1,
            stream_id=0,
            flow_present=True,
            flow_consistent=True,
            flow_bbox=(10.0, 20.0, 50.0, 60.0),
            person_count=2,
            person_bboxes=((1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0)),
            all_detections=(),
            state="YOLO_VERIFY",
            alarm_triggered=True,
            alarm_reason="iou_match",
        )

        d = record.to_dict()
        assert d["frame_index"] == 1
        assert d["flow_present"] is True
        assert d["flow_consistent"] is True
        assert d["flow_bbox"] == [10.0, 20.0, 50.0, 60.0]
        assert d["person_count"] == 2
        assert d["state"] == "YOLO_VERIFY"
        assert d["alarm_triggered"] is True
        assert d["alarm_reason"] == "iou_match"

    def test_dual_probe_summary_to_dict(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        
        from run_flowsentry_dual_probe import DualProbeSummary

        summary = DualProbeSummary(
            frames_processed=100,
            frames_with_flow=50,
            frames_with_persons=30,
            frames_with_alarm=5,
            total_persons=45,
            alarm_triggered=True,
            first_alarm_frame=42,
            stream_ids=(0, 1),
        )

        d = summary.to_dict()
        assert d["frames_processed"] == 100
        assert d["frames_with_flow"] == 50
        assert d["frames_with_persons"] == 30
        assert d["frames_with_alarm"] == 5
        assert d["alarm_triggered"] is True
        assert d["first_alarm_frame"] == 42

    def test_summarize_dual_probe_records(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        
        from run_flowsentry_dual_probe import DualProbeRecord, summarize_dual_probe_records

        records = [
            DualProbeRecord(1, 0, True, False, None, 0, (), (), "FLOW_ACTIVE", False, None),
            DualProbeRecord(2, 0, True, True, (10, 20, 30, 40), 1, ((1, 2, 3, 4),), (), "YOLO_VERIFY", False, None),
            DualProbeRecord(3, 0, True, True, (10, 20, 30, 40), 1, ((1, 2, 3, 4),), (), "ALARM", True, "iou_match"),
        ]

        summary = summarize_dual_probe_records(records)

        assert summary.frames_processed == 3
        assert summary.frames_with_flow == 3
        assert summary.frames_with_persons == 2
        assert summary.frames_with_alarm == 1
        assert summary.total_persons == 2
        assert summary.alarm_triggered is True
        assert summary.first_alarm_frame == 3

    def test_write_dual_probe_jsonl_and_summary(self, tmp_path: Path):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        
        from run_flowsentry_dual_probe import (
            DualProbeRecord,
            DualProbeSummary,
            write_dual_probe_jsonl,
            write_dual_probe_summary_json,
        )

        records = [
            DualProbeRecord(1, 0, True, False, None, 0, (), (), "STANDBY", False, None),
            DualProbeRecord(2, 0, True, True, (10, 20, 30, 40), 1, ((1, 2, 3, 4),), (), "YOLO_VERIFY", True, "test"),
        ]

        summary = DualProbeSummary(
            frames_processed=2,
            frames_with_flow=2,
            frames_with_persons=1,
            frames_with_alarm=1,
            total_persons=1,
            alarm_triggered=True,
            first_alarm_frame=2,
            stream_ids=(0,),
        )

        jsonl_path = write_dual_probe_jsonl(records, tmp_path / "dual" / "records.jsonl")
        summary_path = write_dual_probe_summary_json(summary, tmp_path / "dual" / "summary.json")

        assert jsonl_path.exists()
        assert summary_path.exists()

        lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["frame_index"] == 1

        summary_obj = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary_obj["frames_processed"] == 2
        assert summary_obj["alarm_triggered"] is True

    def test_dual_probe_acceptance_summary_and_report(self, tmp_path: Path):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

        from run_flowsentry_dual_probe import (
            DualProbeRecord,
            evaluate_dual_probe_acceptance,
            summarize_dual_probe_records,
            write_dual_probe_acceptance_report,
        )

        records = [
            DualProbeRecord(1, 0, True, True, (1, 1, 10, 10), 0, (), (), "FLOW_ACTIVE", False, None),
            DualProbeRecord(
                2,
                0,
                True,
                True,
                (1, 1, 10, 10),
                1,
                ((2, 2, 12, 12),),
                ({"class_name": "person", "confidence": 0.9},),
                "ALARM",
                True,
                "person_iou_match",
            ),
        ]
        summary = summarize_dual_probe_records(records)
        video_path = tmp_path / "overlay.mp4"
        video_path.write_bytes(b"x")
        acceptance = evaluate_dual_probe_acceptance(
            records,
            summary,
            preflight_ok=True,
            overlay_video_path=video_path,
            require_overlay=True,
            require_alarm=True,
        )
        assert acceptance.frames_with_joint_bboxes == 1
        assert acceptance.first_joint_bbox_frame == 2
        assert acceptance.passed is True

        report_path = write_dual_probe_acceptance_report(
            acceptance,
            tmp_path / "acceptance_report.md",
            flow_network="flow-net",
            yolo_network="yolo-net",
            source="fakevideo",
            summary=summary,
            jsonl_path=tmp_path / "dual.jsonl",
            summary_json_path=tmp_path / "summary.json",
            overlay_video_path=video_path,
        )
        body = report_path.read_text(encoding="utf-8")
        assert "联合链路验收报告" in body
        assert "flow-net" in body
        assert "通过" in body


class TestDualProbeDisplayFiltering:
    def test_filter_detections_by_flow_bboxes_keeps_center_inside_only(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from run_flowsentry_dual_probe import _filter_detections_by_flow_bboxes

        flow_bboxes = (
            (0.0, 0.0, 100.0, 100.0),
            (200.0, 200.0, 260.0, 260.0),
        )
        detections = (
            SimpleNamespace(
                bbox_xyxy=(10.0, 10.0, 30.0, 30.0),
                class_name="person",
                confidence=0.9,
            ),
            SimpleNamespace(
                bbox_xyxy=(80.0, 80.0, 180.0, 180.0),  # overlaps flow box, but center outside.
                class_name="tv",
                confidence=0.8,
            ),
            SimpleNamespace(
                bbox_xyxy=(210.0, 210.0, 240.0, 240.0),
                class_name="bottle",
                confidence=0.7,
            ),
        )

        kept = _filter_detections_by_flow_bboxes(detections, flow_bboxes)
        assert len(kept) == 2
        assert {d.class_name for d in kept} == {"person", "bottle"}

    def test_prefer_raw_flow_network_switches_from_legacy_alias(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from run_flowsentry_dual_probe import _prefer_raw_flow_network

        resolved = _prefer_raw_flow_network(
            "edgeflownet-opticalflow",
            ("edgeflownet-opticalflow-raw", "yolov8s-coco"),
        )
        assert resolved == "edgeflownet-opticalflow-raw"


class _FakeImage:
    def __init__(self, data):
        self._data = data

    def asarray(self):
        return self._data


@dataclass
class _FakeFlowTensor:
    data: np.ndarray

    def numpy(self):
        return self.data


@dataclass
class _FakeDet:
    box: tuple[float, float, float, float]
    score: float
    class_id: int
    
    @property
    def label(self):
        class _Label:
            name = "person"
        return _Label()


@dataclass
class _FakeFlowFrameResult:
    tensor: _FakeFlowTensor
    image: _FakeImage
    stream_id: int = 0
    src_timestamp: float | None = None


@dataclass
class _FakeYoloFrameResult:
    detections: list[_FakeDet]
    image: _FakeImage
    stream_id: int = 0
    src_timestamp: float | None = None


class _FakeDualStream:
    def __init__(self, frames):
        self._frames = list(frames)
        self.stopped = False

    def __iter__(self):
        return iter(self._frames)

    def stop(self):
        self.stopped = True


class _CaptureOverlayRenderer:
    def __init__(self):
        self.frame_diff_flags: list[bool] = []

    def render(self, frame, signals, _result):
        self.frame_diff_flags.append(bool(signals.frame_diff_triggered))
        return frame


class TestRunDualProbe:
    def test_run_dual_probe_extracts_records(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        
        from run_flowsentry_dual_probe import run_dual_probe

        fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
        fake_flow = np.zeros((100, 100, 2), dtype=np.float32)

        flow_frames = [
            _FakeFlowFrameResult(
                _FakeFlowTensor(fake_flow),
                _FakeImage(fake_image),
                stream_id=0,
            ),
            _FakeFlowFrameResult(
                _FakeFlowTensor(fake_flow),
                _FakeImage(fake_image),
                stream_id=0,
            ),
        ]

        yolo_frames = [
            _FakeYoloFrameResult(
                [_FakeDet((10, 10, 50, 50), 0.9, 0)],
                _FakeImage(fake_image),
                stream_id=0,
            ),
            _FakeYoloFrameResult(
                [_FakeDet((20, 20, 60, 60), 0.8, 0)],
                _FakeImage(fake_image),
                stream_id=0,
            ),
        ]

        call_count = [0]

        def _factory(**kwargs):
            network = kwargs.get("network", "")
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                return _FakeDualStream(flow_frames)
            else:
                return _FakeDualStream(yolo_frames)

        records = run_dual_probe(
            flow_network="edgeflownet-opticalflow",
            yolo_network="yolov8s-coco",
            source="fakevideo",
            max_frames=2,
            stream_factory=_factory,
        )

        assert len(records) == 2
        assert records[0].person_count == 1
        assert records[0].flow_present is False
        assert records[1].person_count == 1

    def test_run_dual_probe_aligns_by_src_timestamp(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

        from run_flowsentry_dual_probe import run_dual_probe

        fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
        fake_flow = np.zeros((100, 100, 2), dtype=np.float32)

        flow_frames = [
            _FakeFlowFrameResult(_FakeFlowTensor(fake_flow), _FakeImage(fake_image), stream_id=0, src_timestamp=1.0),
            _FakeFlowFrameResult(_FakeFlowTensor(fake_flow), _FakeImage(fake_image), stream_id=0, src_timestamp=2.0),
            _FakeFlowFrameResult(_FakeFlowTensor(fake_flow), _FakeImage(fake_image), stream_id=0, src_timestamp=3.0),
        ]
        yolo_frames = [
            _FakeYoloFrameResult([_FakeDet((10, 10, 50, 50), 0.9, 0)], _FakeImage(fake_image), stream_id=0, src_timestamp=2.0),
            _FakeYoloFrameResult([_FakeDet((20, 20, 60, 60), 0.9, 0)], _FakeImage(fake_image), stream_id=0, src_timestamp=3.0),
            _FakeYoloFrameResult([_FakeDet((30, 30, 70, 70), 0.9, 0)], _FakeImage(fake_image), stream_id=0, src_timestamp=4.0),
        ]

        call_count = [0]

        def _factory(**_kwargs):
            call_count[0] += 1
            return _FakeDualStream(flow_frames if call_count[0] % 2 == 1 else yolo_frames)

        records = run_dual_probe(
            flow_network="edgeflownet-opticalflow",
            yolo_network="yolov8s-coco",
            source="fakevideo",
            max_frames=10,
            stream_factory=_factory,
        )

        # Flow@1.0 should be dropped as stale; aligned pairs are (2.0,2.0) and (3.0,3.0).
        assert len(records) == 2
        assert [r.frame_index for r in records] == [1, 2]
        assert all(r.person_count == 1 for r in records)

    def test_run_dual_probe_drops_stale_pairs_by_frame_age(self, monkeypatch):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

        import run_flowsentry_dual_probe as dual_probe_script

        fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
        fake_flow = np.zeros((100, 100, 2), dtype=np.float32)

        flow_frames = [
            _FakeFlowFrameResult(_FakeFlowTensor(fake_flow), _FakeImage(fake_image), stream_id=0, src_timestamp=8.0),
            _FakeFlowFrameResult(_FakeFlowTensor(fake_flow), _FakeImage(fake_image), stream_id=0, src_timestamp=9.8),
        ]
        yolo_frames = [
            _FakeYoloFrameResult([_FakeDet((10, 10, 50, 50), 0.9, 0)], _FakeImage(fake_image), stream_id=0, src_timestamp=8.0),
            _FakeYoloFrameResult([_FakeDet((20, 20, 60, 60), 0.9, 0)], _FakeImage(fake_image), stream_id=0, src_timestamp=9.8),
        ]

        call_count = [0]

        def _factory(**_kwargs):
            call_count[0] += 1
            return _FakeDualStream(flow_frames if call_count[0] % 2 == 1 else yolo_frames)

        monkeypatch.setattr(dual_probe_script.time, "time", lambda: 10.0)
        records = dual_probe_script.run_dual_probe(
            flow_network="edgeflownet-opticalflow",
            yolo_network="yolov8s-coco",
            source="fakevideo",
            max_frames=10,
            stream_factory=_factory,
            max_frame_age_s=0.5,
        )

        # Pair at src_timestamp=8.0 is stale (age=2.0s) and dropped;
        # pair at src_timestamp=9.8 is fresh (age=0.2s) and kept.
        assert len(records) == 1
        assert records[0].frame_index == 1

    def test_run_dual_probe_extracts_once_per_frame_pair(self, monkeypatch):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

        import flowsentry.runtime.adapters as adapters_module
        from run_flowsentry_dual_probe import run_dual_probe

        fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
        fake_flow = np.zeros((100, 100, 2), dtype=np.float32)

        flow_frames = [
            _FakeFlowFrameResult(_FakeFlowTensor(fake_flow), _FakeImage(fake_image), stream_id=0),
            _FakeFlowFrameResult(_FakeFlowTensor(fake_flow), _FakeImage(fake_image), stream_id=0),
            _FakeFlowFrameResult(_FakeFlowTensor(fake_flow), _FakeImage(fake_image), stream_id=0),
        ]
        yolo_frames = [
            _FakeYoloFrameResult([_FakeDet((10, 10, 50, 50), 0.9, 0)], _FakeImage(fake_image), stream_id=0),
            _FakeYoloFrameResult([_FakeDet((20, 20, 60, 60), 0.9, 0)], _FakeImage(fake_image), stream_id=0),
            _FakeYoloFrameResult([_FakeDet((30, 30, 70, 70), 0.9, 0)], _FakeImage(fake_image), stream_id=0),
        ]

        call_count = [0]

        def _factory(**_kwargs):
            call_count[0] += 1
            return _FakeDualStream(flow_frames if call_count[0] % 2 == 1 else yolo_frames)

        captured: dict[str, Any] = {}

        class _CountingFlowBackend:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                self.extract_calls = 0
                captured["flow"] = self

            def extract(self, _frame_result):
                self.extract_calls += 1
                return SimpleNamespace(
                    flow_regions=(),
                    flow_bbox=None,
                    flow_present=False,
                    flow_consistent=False,
                )

        class _CountingYoloBackend:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                self.extract_calls = 0
                captured["yolo"] = self

            def extract(self, _frame_result):
                self.extract_calls += 1
                return SimpleNamespace(
                    detections=(),
                    person_bboxes=(),
                    all_detections=(),
                )

        monkeypatch.setattr(adapters_module, "AxeleraFlowBackend", _CountingFlowBackend)
        monkeypatch.setattr(adapters_module, "AxeleraYoloBackend", _CountingYoloBackend)

        records = run_dual_probe(
            flow_network="edgeflownet-opticalflow",
            yolo_network="yolov8s-coco",
            source="fakevideo",
            max_frames=3,
            stream_factory=_factory,
        )

        assert len(records) == 3
        assert captured["flow"].extract_calls == 3
        assert captured["yolo"].extract_calls == 3

    def test_run_dual_probe_with_overlay_renders_frames(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        
        from flowsentry.overlay import OverlayConfig, OverlayRenderer
        from run_flowsentry_dual_probe import run_dual_probe

        fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
        fake_flow = np.zeros((100, 100, 2), dtype=np.float32)

        flow_frames = [
            _FakeFlowFrameResult(
                _FakeFlowTensor(fake_flow),
                _FakeImage(fake_image),
                stream_id=0,
            ),
        ]

        yolo_frames = [
            _FakeYoloFrameResult(
                [_FakeDet((10, 10, 50, 50), 0.9, 0)],
                _FakeImage(fake_image),
                stream_id=0,
            ),
        ]

        call_count = [0]

        def _factory(**kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                return _FakeDualStream(flow_frames)
            else:
                return _FakeDualStream(yolo_frames)

        renderer = OverlayRenderer(OverlayConfig())

        records = run_dual_probe(
            flow_network="edgeflownet-opticalflow",
            yolo_network="yolov8s-coco",
            source="fakevideo",
            max_frames=1,
            stream_factory=_factory,
            overlay_renderer=renderer,
        )

        assert len(records) == 1
        assert records[0].person_count == 1

    def test_run_dual_probe_uses_frame_diff_to_reach_alarm_path(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

        from flowsentry.config import TriageConfig
        from run_flowsentry_dual_probe import run_dual_probe

        frame0 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame1 = np.full((100, 100, 3), 255, dtype=np.uint8)
        frame2 = np.full((100, 100, 3), 255, dtype=np.uint8)
        flow_motion = np.ones((100, 100, 2), dtype=np.float32) * 10.0

        flow_frames = [
            _FakeFlowFrameResult(_FakeFlowTensor(flow_motion), _FakeImage(frame0), stream_id=0),
            _FakeFlowFrameResult(_FakeFlowTensor(flow_motion), _FakeImage(frame1), stream_id=0),
            _FakeFlowFrameResult(_FakeFlowTensor(flow_motion), _FakeImage(frame2), stream_id=0),
        ]
        yolo_frames = [
            _FakeYoloFrameResult([_FakeDet((10, 10, 90, 90), 0.95, 0)], _FakeImage(frame0), stream_id=0),
            _FakeYoloFrameResult([_FakeDet((10, 10, 90, 90), 0.95, 0)], _FakeImage(frame1), stream_id=0),
            _FakeYoloFrameResult([_FakeDet((10, 10, 90, 90), 0.95, 0)], _FakeImage(frame2), stream_id=0),
        ]

        call_count = [0]

        def _factory(**_kwargs):
            call_count[0] += 1
            return _FakeDualStream(flow_frames if call_count[0] % 2 == 1 else yolo_frames)

        cfg = TriageConfig()
        cfg.flow.consistency_frames_threshold = 1

        records = run_dual_probe(
            flow_network="edgeflownet-opticalflow",
            yolo_network="yolov8s-coco",
            source="fakevideo",
            max_frames=3,
            triage_config=cfg,
            stream_factory=_factory,
        )

        assert [r.state for r in records] == ["STANDBY", "FLOW_ACTIVE", "ALARM"]
        assert records[-1].alarm_triggered is True
        assert records[-1].alarm_reason == "person_iou_match"

    def test_run_dual_probe_passes_frame_diff_signal_to_overlay(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

        from run_flowsentry_dual_probe import run_dual_probe

        frame0 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame1 = np.full((100, 100, 3), 255, dtype=np.uint8)
        flow_motion = np.ones((100, 100, 2), dtype=np.float32) * 10.0

        flow_frames = [
            _FakeFlowFrameResult(_FakeFlowTensor(flow_motion), _FakeImage(frame0), stream_id=0),
            _FakeFlowFrameResult(_FakeFlowTensor(flow_motion), _FakeImage(frame1), stream_id=0),
        ]
        yolo_frames = [
            _FakeYoloFrameResult([_FakeDet((10, 10, 90, 90), 0.95, 0)], _FakeImage(frame0), stream_id=0),
            _FakeYoloFrameResult([_FakeDet((10, 10, 90, 90), 0.95, 0)], _FakeImage(frame1), stream_id=0),
        ]

        call_count = [0]

        def _factory(**_kwargs):
            call_count[0] += 1
            return _FakeDualStream(flow_frames if call_count[0] % 2 == 1 else yolo_frames)

        renderer = _CaptureOverlayRenderer()
        run_dual_probe(
            flow_network="edgeflownet-opticalflow",
            yolo_network="yolov8s-coco",
            source="fakevideo",
            max_frames=2,
            stream_factory=_factory,
            overlay_renderer=renderer,
        )

        assert renderer.frame_diff_flags == [False, True]

    def test_run_dual_probe_notifier_called_only_on_alarm_frames(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from flowsentry.config import TriageConfig
        from run_flowsentry_dual_probe import run_dual_probe

        frame0 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame1 = np.full((100, 100, 3), 255, dtype=np.uint8)
        frame2 = np.full((100, 100, 3), 255, dtype=np.uint8)
        flow_motion = np.ones((100, 100, 2), dtype=np.float32) * 10.0

        flow_frames = [
            _FakeFlowFrameResult(_FakeFlowTensor(flow_motion), _FakeImage(frame0), stream_id=0),
            _FakeFlowFrameResult(_FakeFlowTensor(flow_motion), _FakeImage(frame1), stream_id=0),
            _FakeFlowFrameResult(_FakeFlowTensor(flow_motion), _FakeImage(frame2), stream_id=0),
        ]
        yolo_frames = [
            _FakeYoloFrameResult([_FakeDet((10, 10, 90, 90), 0.95, 0)], _FakeImage(frame0), stream_id=0),
            _FakeYoloFrameResult([_FakeDet((10, 10, 90, 90), 0.95, 0)], _FakeImage(frame1), stream_id=0),
            _FakeYoloFrameResult([_FakeDet((10, 10, 90, 90), 0.95, 0)], _FakeImage(frame2), stream_id=0),
        ]

        call_count = [0]

        def _factory(**_kwargs):
            call_count[0] += 1
            return _FakeDualStream(flow_frames if call_count[0] % 2 == 1 else yolo_frames)

        class _CaptureNotifier:
            def __init__(self):
                self.events = []

            def notify_if_needed(self, event):
                self.events.append(event)
                return SimpleNamespace(sent=False, reason="captured", status_code=None, error=None)

        notifier = _CaptureNotifier()
        cfg = TriageConfig()
        cfg.flow.consistency_frames_threshold = 1

        records = run_dual_probe(
            flow_network="edgeflownet-opticalflow",
            yolo_network="yolov8s-coco",
            source="fakevideo",
            max_frames=3,
            triage_config=cfg,
            stream_factory=_factory,
            alarm_notifier=notifier,
        )

        assert [r.alarm_triggered for r in records] == [False, False, True]
        assert len(notifier.events) == 1
        assert notifier.events[0].alarm_reason == "person_iou_match"
