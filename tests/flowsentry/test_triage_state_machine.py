# tests/flowsentry/test_triage_state_machine.py
# Copyright 2025, FlowSentry-Wake
# 扩展版本

import pytest
from flowsentry.fsm.triage_state_machine import TriageState, TriageStateMachine, TriageStepResult


# === 原有测试（保持不变）===

def test_state_machine_basic_progression_to_alarm():
    fsm = TriageStateMachine(no_motion_reset_frames=2)

    step1 = fsm.step(
        frame_diff_triggered=True,
        flow_present=True,
        flow_threshold_reached=False,
        alarm_triggered=False,
    )
    assert step1.state == TriageState.FLOW_ACTIVE
    assert step1.optical_flow_enabled is True
    assert step1.yolo_enabled is False

    step2 = fsm.step(
        frame_diff_triggered=False,
        flow_present=True,
        flow_threshold_reached=True,
        alarm_triggered=True,
    )
    assert step2.state == TriageState.ALARM
    assert step2.optical_flow_enabled is True
    assert step2.yolo_enabled is True


def test_state_machine_resets_to_standby_after_no_motion_timeout():
    fsm = TriageStateMachine(no_motion_reset_frames=2)

    fsm.step(
        frame_diff_triggered=True,
        flow_present=True,
        flow_threshold_reached=False,
        alarm_triggered=False,
    )
    fsm.step(
        frame_diff_triggered=False,
        flow_present=False,
        flow_threshold_reached=False,
        alarm_triggered=False,
    )
    step = fsm.step(
        frame_diff_triggered=False,
        flow_present=False,
        flow_threshold_reached=False,
        alarm_triggered=False,
    )

    assert step.state == TriageState.STANDBY
    assert step.optical_flow_enabled is False


# === 新增测试 ===

class TestTriageStateMachine:
    def test_initial_state_is_standby(self):
        """初始状态为 STANDBY"""
        fsm = TriageStateMachine()
        assert fsm.state == TriageState.STANDBY

    def test_invalid_reset_frames_raises(self):
        """无效的 no_motion_reset_frames 抛出异常"""
        with pytest.raises(ValueError, match="no_motion_reset_frames must be >= 1"):
            TriageStateMachine(no_motion_reset_frames=0)

    def test_no_trigger_stays_in_standby(self):
        """无触发时保持在 STANDBY"""
        fsm = TriageStateMachine()
        result = fsm.step(
            frame_diff_triggered=False,
            flow_present=False,
            flow_threshold_reached=False,
            alarm_triggered=False,
        )
        assert result.state == TriageState.STANDBY
        assert result.transitioned is False

    def test_standby_to_flow_active_on_frame_diff(self):
        """帧差触发后进入 FLOW_ACTIVE"""
        fsm = TriageStateMachine()
        result = fsm.step(
            frame_diff_triggered=True,
            flow_present=True,
            flow_threshold_reached=False,
            alarm_triggered=False,
        )
        assert result.state == TriageState.FLOW_ACTIVE
        assert result.transitioned is True
        assert result.optical_flow_enabled is True
        assert result.yolo_enabled is False

    def test_flow_active_to_yolo_verify_without_alarm(self):
        """FLOW_ACTIVE 阈值达到但无报警时进入 YOLO_VERIFY"""
        fsm = TriageStateMachine()
        fsm.step(
            frame_diff_triggered=True,
            flow_present=True,
            flow_threshold_reached=False,
            alarm_triggered=False,
        )
        result = fsm.step(
            frame_diff_triggered=False,
            flow_present=True,
            flow_threshold_reached=True,
            alarm_triggered=False,
        )
        assert result.state == TriageState.YOLO_VERIFY
        assert result.yolo_enabled is True

    def test_yolo_verify_to_alarm(self):
        """YOLO_VERIFY 检测到报警进入 ALARM"""
        fsm = TriageStateMachine()
        fsm.step(frame_diff_triggered=True, flow_present=True, flow_threshold_reached=False, alarm_triggered=False)
        fsm.step(frame_diff_triggered=False, flow_present=True, flow_threshold_reached=True, alarm_triggered=False)
        result = fsm.step(
            frame_diff_triggered=False,
            flow_present=True,
            flow_threshold_reached=True,
            alarm_triggered=True,
        )
        assert result.state == TriageState.ALARM

    def test_reset_clears_state(self):
        """reset() 清除状态"""
        fsm = TriageStateMachine()
        fsm.step(frame_diff_triggered=True, flow_present=True, flow_threshold_reached=False, alarm_triggered=False)
        assert fsm.state == TriageState.FLOW_ACTIVE
        fsm.reset()
        assert fsm.state == TriageState.STANDBY
        assert fsm.no_motion_frames == 0

    def test_no_motion_frames_counter(self):
        """无运动帧计数正确"""
        fsm = TriageStateMachine(no_motion_reset_frames=3)
        fsm.step(frame_diff_triggered=True, flow_present=True, flow_threshold_reached=False, alarm_triggered=False)

        r1 = fsm.step(frame_diff_triggered=False, flow_present=False, flow_threshold_reached=False, alarm_triggered=False)
        assert r1.no_motion_frames == 1

        r2 = fsm.step(frame_diff_triggered=False, flow_present=False, flow_threshold_reached=False, alarm_triggered=False)
        assert r2.no_motion_frames == 2

        r3 = fsm.step(frame_diff_triggered=False, flow_present=True, flow_threshold_reached=False, alarm_triggered=False)
        assert r3.no_motion_frames == 0

    def test_flow_active_resets_on_idle_timeout(self):
        """FLOW_ACTIVE 空闲超时后回到 STANDBY"""
        fsm = TriageStateMachine(no_motion_reset_frames=2)
        fsm.step(frame_diff_triggered=True, flow_present=True, flow_threshold_reached=False, alarm_triggered=False)
        fsm.step(frame_diff_triggered=False, flow_present=False, flow_threshold_reached=False, alarm_triggered=False)
        result = fsm.step(frame_diff_triggered=False, flow_present=False, flow_threshold_reached=False, alarm_triggered=False)
        assert result.state == TriageState.STANDBY

    def test_alarm_resets_on_idle_timeout(self):
        """ALARM 空闲超时后回到 STANDBY"""
        fsm = TriageStateMachine(no_motion_reset_frames=2)
        fsm.step(frame_diff_triggered=True, flow_present=True, flow_threshold_reached=False, alarm_triggered=False)
        fsm.step(frame_diff_triggered=False, flow_present=True, flow_threshold_reached=True, alarm_triggered=True)
        assert fsm.state == TriageState.ALARM
        fsm.step(frame_diff_triggered=False, flow_present=False, flow_threshold_reached=False, alarm_triggered=False)
        result = fsm.step(frame_diff_triggered=False, flow_present=False, flow_threshold_reached=False, alarm_triggered=False)
        assert result.state == TriageState.STANDBY

    def test_step_result_frozen(self):
        """TriageStepResult 是不可变的"""
        fsm = TriageStateMachine()
        result = fsm.step(frame_diff_triggered=False, flow_present=False, flow_threshold_reached=False, alarm_triggered=False)
        with pytest.raises(Exception):
            result.state = TriageState.ALARM


class TestTriageState:
    def test_state_enum_values(self):
        """状态枚举值正确"""
        assert TriageState.STANDBY.value == "standby"
        assert TriageState.FLOW_ACTIVE.value == "flow_active"
        assert TriageState.YOLO_VERIFY.value == "yolo_verify"
        assert TriageState.ALARM.value == "alarm"
