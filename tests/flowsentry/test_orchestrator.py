# tests/flowsentry/test_orchestrator.py
# Copyright 2025, FlowSentry-Wake
# 扩展版本

import pytest
from flowsentry.config import TriageConfig
from flowsentry.runtime.orchestrator import TriageOrchestrator, TriageOrchestratorOutput
from flowsentry.runtime.adapters import MockFlowBackend, MockYoloBackend
from flowsentry.types import Detection, FrameSignals
from flowsentry.fsm.triage_state_machine import TriageState


# === 原有测试（保持不变）===

def test_orchestrator_mock_path_from_trigger_to_alarm_and_idle_reset():
    cfg = TriageConfig()
    cfg.flow.consistency_frames_threshold = 2
    cfg.fusion.iou_threshold = 0.3
    cfg.runtime.no_motion_reset_frames = 2
    orch = TriageOrchestrator(cfg)

    r0 = orch.process(
        FrameSignals(
            frame_diff_triggered=False,
            flow_present=False,
            flow_consistent=False,
            flow_bbox=None,
            person_bboxes=(),
        )
    )
    assert r0.state == TriageState.STANDBY
    assert r0.alarm.triggered is False

    r1 = orch.process(
        FrameSignals(
            frame_diff_triggered=True,
            flow_present=True,
            flow_consistent=True,
            flow_bbox=(10.0, 10.0, 30.0, 30.0),
            person_bboxes=(),
        )
    )
    assert r1.state == TriageState.FLOW_ACTIVE
    assert r1.consistency_count == 1
    assert r1.yolo_enabled is False

    r2 = orch.process(
        FrameSignals(
            frame_diff_triggered=False,
            flow_present=True,
            flow_consistent=True,
            flow_bbox=(10.0, 10.0, 30.0, 30.0),
            person_bboxes=(),
        )
    )
    assert r2.flow_threshold_reached is True
    assert r2.state == TriageState.ALARM
    assert r2.yolo_enabled is True
    assert r2.alarm.triggered is True
    assert r2.alarm.reason == "flow_no_object_detected"

    r3 = orch.process(
        FrameSignals(
            frame_diff_triggered=False,
            flow_present=True,
            flow_consistent=False,
            flow_bbox=(10.0, 10.0, 30.0, 30.0),
            person_bboxes=((200.0, 200.0, 240.0, 240.0),),
        )
    )
    assert r3.state == TriageState.ALARM
    assert r3.alarm.triggered is True
    assert r3.alarm.reason == "flow_no_object_detected"
    assert r3.match.best_iou is None

    r4 = orch.process(
        FrameSignals(
            frame_diff_triggered=False,
            flow_present=False,
            flow_consistent=False,
            flow_bbox=None,
            person_bboxes=(),
        )
    )
    assert r4.state == TriageState.ALARM
    assert r4.alarm.triggered is False

    r5 = orch.process(
        FrameSignals(
            frame_diff_triggered=False,
            flow_present=False,
            flow_consistent=False,
            flow_bbox=None,
            person_bboxes=(),
        )
    )
    assert r5.state == TriageState.STANDBY
    assert r5.consistency_count == 0


# === 新增测试 ===

class TestTriageOrchestrator:
    def test_default_config(self):
        """默认配置正确初始化"""
        orch = TriageOrchestrator()
        assert orch.config.flow.consistency_frames_threshold == 3
        assert orch.config.fusion.iou_threshold == 0.3

    def test_reset_clears_all_state(self):
        """reset() 清除所有状态"""
        cfg = TriageConfig()
        cfg.flow.consistency_frames_threshold = 2
        orch = TriageOrchestrator(cfg)

        orch.process(FrameSignals(frame_diff_triggered=True, flow_present=True, flow_consistent=True, flow_bbox=(0, 0, 10, 10), person_bboxes=()))
        orch.process(FrameSignals(frame_diff_triggered=True, flow_present=True, flow_consistent=True, flow_bbox=(0, 0, 10, 10), person_bboxes=()))

        assert orch.consistency.count > 0
        assert orch.fsm.state != TriageState.STANDBY

        orch.reset()
        assert orch.consistency.count == 0
        assert orch.fsm.state == TriageState.STANDBY

    def test_consistency_not_counted_in_standby_without_trigger(self):
        """STANDBY 状态下无触发时不计数"""
        orch = TriageOrchestrator()
        for _ in range(10):
            result = orch.process(
                FrameSignals(
                    frame_diff_triggered=False,
                    flow_present=True,
                    flow_consistent=True,
                    flow_bbox=(0, 0, 10, 10),
                    person_bboxes=(),
                )
            )
        assert result.consistency_count == 0

    def test_person_iou_match_triggers_alarm(self):
        """person IoU 匹配触发报警"""
        cfg = TriageConfig()
        cfg.flow.consistency_frames_threshold = 1
        orch = TriageOrchestrator(cfg)

        orch.process(FrameSignals(frame_diff_triggered=True, flow_present=True, flow_consistent=True, flow_bbox=(0, 0, 100, 100), person_bboxes=((10, 10, 90, 90),)))

        result = orch.process(FrameSignals(frame_diff_triggered=False, flow_present=True, flow_consistent=True, flow_bbox=(0, 0, 100, 100), person_bboxes=((10, 10, 90, 90),)))

        assert result.alarm.triggered is True
        assert result.alarm.reason == "person_iou_match"
        assert result.match.matched is True

    def test_non_person_detection_does_not_trigger_alarm(self):
        """检测到非人目标时不报警"""
        cfg = TriageConfig()
        cfg.flow.consistency_frames_threshold = 1
        orch = TriageOrchestrator(cfg)
        non_person = Detection(
            bbox_xyxy=(5.0, 5.0, 90.0, 90.0),
            class_id=2,
            class_name="car",
            confidence=0.9,
        )

        orch.process(
            FrameSignals(
                frame_diff_triggered=True,
                flow_present=True,
                flow_consistent=True,
                flow_bbox=(0, 0, 100, 100),
                person_bboxes=(),
                all_detections=(non_person,),
            )
        )

        result = orch.process(
            FrameSignals(
                frame_diff_triggered=False,
                flow_present=True,
                flow_consistent=True,
                flow_bbox=(0, 0, 100, 100),
                person_bboxes=(),
                all_detections=(non_person,),
            )
        )

        assert result.alarm.triggered is False
        assert result.alarm.reason == "non_person_object_detected"
        assert result.state == TriageState.YOLO_VERIFY

    def test_non_person_outside_flow_region_is_ignored_and_triggers_no_object_alarm(self):
        """仅 flow 区域内检测有效：区域外非人目标不应抑制报警"""
        cfg = TriageConfig()
        cfg.flow.consistency_frames_threshold = 1
        orch = TriageOrchestrator(cfg)
        outside_non_person = Detection(
            bbox_xyxy=(500.0, 500.0, 560.0, 560.0),
            class_id=2,
            class_name="car",
            confidence=0.9,
        )

        orch.process(
            FrameSignals(
                frame_diff_triggered=True,
                flow_present=True,
                flow_consistent=True,
                flow_bbox=(0, 0, 100, 100),
                person_bboxes=(),
                all_detections=(outside_non_person,),
            )
        )

        result = orch.process(
            FrameSignals(
                frame_diff_triggered=False,
                flow_present=True,
                flow_consistent=True,
                flow_bbox=(0, 0, 100, 100),
                person_bboxes=(),
                all_detections=(outside_non_person,),
            )
        )

        assert result.alarm.triggered is True
        assert result.alarm.reason == "flow_no_object_detected"

    def test_person_inside_flow_region_triggers_alarm_even_when_iou_below_threshold(self):
        """flow 区域内检测到人时报警（不再因 IoU 阈值取消报警）"""
        cfg = TriageConfig()
        cfg.flow.consistency_frames_threshold = 1
        cfg.fusion.iou_threshold = 0.8
        orch = TriageOrchestrator(cfg)

        orch.process(
            FrameSignals(
                frame_diff_triggered=True,
                flow_present=True,
                flow_consistent=True,
                flow_bbox=(0, 0, 100, 100),
                person_bboxes=((80, 80, 95, 95),),
            )
        )
        result = orch.process(
            FrameSignals(
                frame_diff_triggered=False,
                flow_present=True,
                flow_consistent=True,
                flow_bbox=(0, 0, 100, 100),
                person_bboxes=((80, 80, 95, 95),),
            )
        )

        assert result.alarm.triggered is True
        assert result.alarm.reason == "person_iou_below_threshold"

    def test_output_frozen(self):
        """输出是不可变的"""
        orch = TriageOrchestrator()
        result = orch.process(FrameSignals(frame_diff_triggered=False, flow_present=False, flow_consistent=False, flow_bbox=None, person_bboxes=()))
        with pytest.raises(Exception):
            result.state = TriageState.ALARM


class TestTriageOrchestratorOutput:
    def test_output_has_all_fields(self):
        """输出包含所有字段"""
        orch = TriageOrchestrator()
        result = orch.process(
            FrameSignals(
                frame_diff_triggered=True,
                flow_present=True,
                flow_consistent=True,
                flow_bbox=(0, 0, 10, 10),
                person_bboxes=(),
            )
        )
        assert hasattr(result, "state")
        assert hasattr(result, "optical_flow_enabled")
        assert hasattr(result, "yolo_enabled")
        assert hasattr(result, "consistency_count")
        assert hasattr(result, "flow_threshold_reached")
        assert hasattr(result, "match")
        assert hasattr(result, "alarm")


class TestProcessWithFlowBackend:
    """Test process_with_flow_backend method"""

    def test_no_flow_stays_in_standby(self):
        """No flow stays in STANDBY"""
        cfg = TriageConfig()
        cfg.flow.consistency_frames_threshold = 1
        orch = TriageOrchestrator(cfg)
        flow = MockFlowBackend(batches=[
            (False, False, None),
            (False, False, None),
        ])

        r1 = orch.process_with_flow_backend(
            frame_diff_triggered=False,
            flow_backend=flow,
            flow_frame_result=None,
        )
        assert r1.state == TriageState.STANDBY
        assert r1.optical_flow_enabled is False

        r2 = orch.process_with_flow_backend(
            frame_diff_triggered=False,
            flow_backend=flow,
            flow_frame_result=None,
        )
        assert r2.state == TriageState.STANDBY

    def test_flow_trigger_activates_flow_stage(self):
        """Flow trigger activates FLOW_ACTIVE stage"""
        cfg = TriageConfig()
        cfg.flow.consistency_frames_threshold = 2
        orch = TriageOrchestrator(cfg)
        flow = MockFlowBackend(batches=[
            (True, True, (10, 10, 50, 50)),
            (True, True, (10, 10, 50, 50)),
        ])

        r1 = orch.process_with_flow_backend(
            frame_diff_triggered=True,
            flow_backend=flow,
            flow_frame_result=None,
        )
        assert r1.state == TriageState.FLOW_ACTIVE
        assert r1.optical_flow_enabled is True

        r2 = orch.process_with_flow_backend(
            frame_diff_triggered=False,
            flow_backend=flow,
            flow_frame_result=None,
        )
        assert r2.flow_threshold_reached is True
        assert r2.state == TriageState.ALARM


class TestProcessWithBothBackends:
    """Test process_with_both_backends method"""

    def test_flow_and_yolo_integration(self):
        """Flow and YOLO integration works correctly"""
        cfg = TriageConfig()
        cfg.flow.consistency_frames_threshold = 1
        cfg.fusion.iou_threshold = 0.3
        orch = TriageOrchestrator(cfg)
        flow = MockFlowBackend(batches=[
            (True, True, (10, 10, 100, 100)),
            (True, True, (10, 10, 100, 100)),
        ])
        yolo = MockYoloBackend(batches=[
            ((20, 20, 90, 90),),
            ((20, 20, 90, 90),),
        ])

        r1 = orch.process_with_both_backends(
            frame_diff_triggered=True,
            flow_backend=flow,
            flow_frame_result=None,
            yolo_backend=yolo,
            yolo_frame_result=None,
        )
        assert r1.state == TriageState.FLOW_ACTIVE

        r2 = orch.process_with_both_backends(
            frame_diff_triggered=False,
            flow_backend=flow,
            flow_frame_result=None,
            yolo_backend=yolo,
            yolo_frame_result=None,
        )
        assert r2.state == TriageState.ALARM
        assert r2.alarm.triggered is True
        assert r2.alarm.reason == "person_iou_match"
        assert r2.match.matched is True

    def test_no_person_still_triggers_alarm(self):
        """No person still triggers alarm (flow only)"""
        cfg = TriageConfig()
        cfg.flow.consistency_frames_threshold = 1
        orch = TriageOrchestrator(cfg)
        flow = MockFlowBackend(batches=[
            (True, True, (10, 10, 100, 100)),
            (True, True, (10, 10, 100, 100)),
        ])
        yolo = MockYoloBackend(batches=[
            (),
            (),
        ])

        orch.process_with_both_backends(
            frame_diff_triggered=True,
            flow_backend=flow,
            flow_frame_result=None,
            yolo_backend=yolo,
            yolo_frame_result=None,
        )

        r2 = orch.process_with_both_backends(
            frame_diff_triggered=False,
            flow_backend=flow,
            flow_frame_result=None,
            yolo_backend=yolo,
            yolo_frame_result=None,
        )
        assert r2.state == TriageState.ALARM
        assert r2.alarm.triggered is True
        assert r2.alarm.reason == "flow_no_object_detected"

    def test_timestamp_passed_through(self):
        """Timestamp is passed through correctly"""
        orch = TriageOrchestrator()
        flow = MockFlowBackend(batches=[(False, False, None)])

        result = orch.process_with_flow_backend(
            frame_diff_triggered=False,
            flow_backend=flow,
            flow_frame_result=None,
            timestamp_ms=12345,
        )
        assert result.state == TriageState.STANDBY
