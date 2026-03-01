# tests/flowsentry/test_types.py
# Copyright 2025, FlowSentry-Wake

import pytest
from flowsentry.types import BBox, Detection, FlowRegion, MotionEvent, AlarmDecision, FrameSignals


class TestDetection:
    def test_detection_creation(self):
        det = Detection(
            bbox_xyxy=(10.0, 20.0, 100.0, 200.0),
            class_id=0,
            class_name="person",
            confidence=0.95,
        )
        assert det.class_name == "person"
        assert det.confidence == 0.95
        assert det.bbox_xyxy == (10.0, 20.0, 100.0, 200.0)

    def test_detection_frozen(self):
        det = Detection(bbox_xyxy=(0, 0, 10, 10), class_id=0, class_name="cat", confidence=0.8)
        with pytest.raises(Exception):
            det.confidence = 0.9


class TestFlowRegion:
    def test_flow_region_creation(self):
        region = FlowRegion(bbox_xyxy=(5.0, 5.0, 50.0, 50.0), area=2025)
        assert region.area == 2025
        assert region.bbox_xyxy == (5.0, 5.0, 50.0, 50.0)


class TestMotionEvent:
    def test_motion_event_not_triggered(self):
        event = MotionEvent(triggered=False, change_ratio=0.005, region_area=100)
        assert event.triggered is False
        assert event.bbox_xyxy is None

    def test_motion_event_triggered_with_bbox(self):
        event = MotionEvent(
            triggered=True,
            change_ratio=0.02,
            region_area=500,
            bbox_xyxy=(10, 10, 50, 50),
        )
        assert event.triggered is True
        assert event.region_area == 500


class TestAlarmDecision:
    def test_alarm_decision_no_flow(self):
        decision = AlarmDecision(triggered=False, reason="no_flow")
        assert decision.triggered is False
        assert decision.reason == "no_flow"

    def test_alarm_decision_with_iou(self):
        decision = AlarmDecision(
            triggered=True,
            reason="person_iou_match",
            best_iou=0.75,
            matched=True,
            flow_bbox=(0, 0, 10, 10),
            person_bbox=(1, 1, 9, 9),
        )
        assert decision.matched is True
        assert decision.best_iou == 0.75


class TestFrameSignals:
    def test_frame_signals_defaults(self):
        signals = FrameSignals(
            frame_diff_triggered=False,
            flow_present=False,
            flow_consistent=False,
        )
        assert signals.person_bboxes == ()
        assert signals.flow_bbox is None
        assert signals.timestamp_ms is None

    def test_frame_signals_full(self):
        signals = FrameSignals(
            frame_diff_triggered=True,
            flow_present=True,
            flow_consistent=True,
            flow_bbox=(10.0, 10.0, 50.0, 50.0),
            person_bboxes=((5.0, 5.0, 55.0, 55.0),),
            timestamp_ms=1234567890,
        )
        assert len(signals.person_bboxes) == 1
        assert signals.timestamp_ms == 1234567890
