from __future__ import annotations

from dataclasses import dataclass

from flowsentry.config import TriageConfig
from flowsentry.runtime.adapters import AxeleraYoloBackend, MockYoloBackend
from flowsentry.runtime.orchestrator import TriageOrchestrator
from flowsentry.fsm.triage_state_machine import TriageState


@dataclass
class _FakeLabel:
    name: str


class _FakeDet:
    def __init__(self, box, score, class_id, label_name=None, supports_is_a=False):
        self.box = box
        self.score = score
        self.class_id = class_id
        self._supports_is_a = supports_is_a
        if label_name is not None:
            self.label = _FakeLabel(label_name)

    def is_a(self, name: str) -> bool:
        if not self._supports_is_a:
            raise AttributeError("is_a not supported")
        return hasattr(self, "label") and self.label.name == name


class _FakeFrameResult:
    def __init__(self, detections):
        self.detections = detections


def test_axelera_yolo_backend_extracts_person_bboxes_and_filters_confidence():
    backend = AxeleraYoloBackend(min_confidence=0.5)
    frame = _FakeFrameResult(
        detections=[
            _FakeDet((1, 2, 20, 30), 0.8, 0, "person"),
            _FakeDet((5, 6, 15, 16), 0.2, 0, "person"),
            _FakeDet((0, 0, 5, 5), 0.95, 2, "car"),
        ]
    )

    out = backend.extract(frame)

    assert len(out.detections) == 1
    assert out.person_bboxes == ((1.0, 2.0, 20.0, 30.0),)


def test_axelera_yolo_backend_supports_is_a_and_class_id_fallback():
    backend = AxeleraYoloBackend(min_confidence=0.1)
    frame = _FakeFrameResult(
        detections=[
            _FakeDet((10, 10, 20, 20), 0.7, 7, "person", supports_is_a=True),
            _FakeDet((30, 30, 50, 50), 0.9, 0, None),  # no label, class_id fallback
        ]
    )

    person_bboxes = backend.extract_person_bboxes(frame)

    assert person_bboxes == (
        (30.0, 30.0, 50.0, 50.0),
        (10.0, 10.0, 20.0, 20.0),
    ) or person_bboxes == (
        (10.0, 10.0, 20.0, 20.0),
        (30.0, 30.0, 50.0, 50.0),
    )


def test_mock_yolo_backend_and_orchestrator_integration():
    cfg = TriageConfig()
    cfg.flow.consistency_frames_threshold = 1
    orch = TriageOrchestrator(cfg)
    yolo = MockYoloBackend([((12.0, 12.0, 28.0, 28.0),)])

    r1 = orch.process_with_yolo_backend(
        frame_diff_triggered=True,
        flow_present=True,
        flow_consistent=True,
        flow_bbox=(10.0, 10.0, 30.0, 30.0),
        yolo_backend=yolo,
        yolo_frame_result=None,
    )

    assert r1.state == TriageState.FLOW_ACTIVE

    r2 = orch.process_with_yolo_backend(
        frame_diff_triggered=False,
        flow_present=True,
        flow_consistent=True,
        flow_bbox=(10.0, 10.0, 30.0, 30.0),
        yolo_backend=MockYoloBackend([((12.0, 12.0, 28.0, 28.0),)]),
        yolo_frame_result=None,
    )

    assert r2.state == TriageState.ALARM
    assert r2.alarm.triggered is True
    assert r2.alarm.reason == "person_iou_match"
