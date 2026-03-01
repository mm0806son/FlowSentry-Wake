# tests/flowsentry/test_alarm_policy.py
# Copyright 2025, FlowSentry-Wake
# 扩展版本

import pytest
from flowsentry.fusion.alarm_policy import AlarmPolicy
from flowsentry.types import Detection


# === 原有测试（保持不变）===

def test_alarm_policy_alarms_on_flow_only_when_no_detections():
    policy = AlarmPolicy(iou_threshold=0.3)
    decision = policy.evaluate(
        flow_present=True,
        flow_bbox=(0.0, 0.0, 10.0, 10.0),
        person_bboxes=(),
        best_iou=None,
        best_person_bbox=None,
    )

    assert decision.triggered is True
    assert decision.reason == "flow_no_object_detected"


def test_alarm_policy_alarms_on_person_match_and_no_alarm_on_mismatch():
    policy = AlarmPolicy(iou_threshold=0.3)
    flow_bbox = (0.0, 0.0, 10.0, 10.0)
    person_bbox = (1.0, 1.0, 9.0, 9.0)

    match_decision = policy.evaluate(
        flow_present=True,
        flow_bbox=flow_bbox,
        person_bboxes=(person_bbox,),
        best_iou=0.7,
        best_person_bbox=person_bbox,
    )
    mismatch_decision = policy.evaluate(
        flow_present=True,
        flow_bbox=flow_bbox,
        person_bboxes=(person_bbox,),
        best_iou=0.1,
        best_person_bbox=person_bbox,
    )

    assert match_decision.triggered is True
    assert match_decision.reason == "person_iou_match"
    assert mismatch_decision.triggered is True
    assert mismatch_decision.reason == "person_iou_below_threshold"


def test_alarm_policy_no_alarm_when_non_person_object_detected():
    policy = AlarmPolicy(iou_threshold=0.3)
    flow_bbox = (0.0, 0.0, 10.0, 10.0)
    non_person = Detection(
        bbox_xyxy=(2.0, 2.0, 8.0, 8.0),
        class_id=2,
        class_name="car",
        confidence=0.9,
    )

    decision = policy.evaluate(
        flow_present=True,
        flow_bbox=flow_bbox,
        person_bboxes=(),
        best_iou=None,
        best_person_bbox=None,
        all_detections=(non_person,),
    )

    assert decision.triggered is False
    assert decision.reason == "non_person_object_detected"


def test_alarm_policy_no_alarm_when_no_flow():
    policy = AlarmPolicy(iou_threshold=0.3)
    decision = policy.evaluate(
        flow_present=False,
        flow_bbox=None,
        person_bboxes=(),
        best_iou=None,
        best_person_bbox=None,
    )

    assert decision.triggered is False
    assert decision.reason == "no_flow"


# === 新增测试 ===

class TestAlarmPolicy:
    def test_flow_present_but_no_bbox_still_alarm_if_no_detections(self):
        """有光流且无检测目标时报警"""
        policy = AlarmPolicy(iou_threshold=0.3)
        decision = policy.evaluate(
            flow_present=True,
            flow_bbox=None,
            person_bboxes=(),
            best_iou=None,
            best_person_bbox=None,
        )
        assert decision.triggered is True

    def test_multiple_persons_with_mixed_iou(self):
        """多个 person 时匹配结果基于 best_iou"""
        policy = AlarmPolicy(iou_threshold=0.5)
        flow_bbox = (0.0, 0.0, 100.0, 100.0)
        persons = ((10.0, 10.0, 90.0, 90.0), (200.0, 200.0, 300.0, 300.0))

        decision_match = policy.evaluate(
            flow_present=True,
            flow_bbox=flow_bbox,
            person_bboxes=persons,
            best_iou=0.7,
            best_person_bbox=persons[0],
        )
        assert decision_match.matched is True
        assert decision_match.reason == "person_iou_match"

        decision_mismatch = policy.evaluate(
            flow_present=True,
            flow_bbox=flow_bbox,
            person_bboxes=persons,
            best_iou=0.2,
            best_person_bbox=persons[1],
        )
        assert decision_mismatch.matched is False
        assert decision_mismatch.reason == "person_iou_below_threshold"
        assert decision_mismatch.triggered is True

    def test_iou_exactly_at_threshold(self):
        """IoU 恰好等于阈值时匹配"""
        policy = AlarmPolicy(iou_threshold=0.5)
        decision = policy.evaluate(
            flow_present=True,
            flow_bbox=(0, 0, 10, 10),
            person_bboxes=((0, 0, 10, 10),),
            best_iou=0.5,
            best_person_bbox=(0, 0, 10, 10),
        )
        assert decision.matched is True
        assert decision.reason == "person_iou_match"

    def test_iou_just_below_threshold(self):
        """IoU 略低于阈值时也报警（但标记为 below_threshold）"""
        policy = AlarmPolicy(iou_threshold=0.5)
        decision = policy.evaluate(
            flow_present=True,
            flow_bbox=(0, 0, 10, 10),
            person_bboxes=((0, 0, 10, 10),),
            best_iou=0.49,
            best_person_bbox=(0, 0, 10, 10),
        )
        assert decision.matched is False
        assert decision.reason == "person_iou_below_threshold"
        assert decision.triggered is True

    def test_decision_contains_all_fields(self):
        """决策包含所有字段"""
        policy = AlarmPolicy(iou_threshold=0.3)
        decision = policy.evaluate(
            flow_present=True,
            flow_bbox=(0.0, 0.0, 10.0, 10.0),
            person_bboxes=((1.0, 1.0, 9.0, 9.0),),
            best_iou=0.8,
            best_person_bbox=(1.0, 1.0, 9.0, 9.0),
        )
        assert decision.flow_bbox == (0.0, 0.0, 10.0, 10.0)
        assert decision.person_bbox == (1.0, 1.0, 9.0, 9.0)
        assert decision.best_iou == 0.8
        assert decision.matched is True
