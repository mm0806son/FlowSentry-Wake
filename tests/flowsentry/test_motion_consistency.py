# tests/flowsentry/test_motion_consistency.py
# Copyright 2025, FlowSentry-Wake
# 扩展版本

import pytest
from flowsentry.motion.consistency import MotionConsistencyCounter


# === 原有测试（保持不变）===

def test_consistency_threshold_reached_only_after_n_frames():
    counter = MotionConsistencyCounter(threshold_frames=3)

    assert counter.update(True) is False
    assert counter.count == 1
    assert counter.update(True) is False
    assert counter.count == 2
    assert counter.update(True) is True
    assert counter.count == 3


def test_consistency_resets_on_false():
    counter = MotionConsistencyCounter(threshold_frames=2)

    assert counter.update(True) is False
    assert counter.update(False) is False
    assert counter.count == 0
    assert counter.update(True) is False
    assert counter.update(True) is True


# === 新增测试 ===

class TestMotionConsistencyCounter:
    def test_threshold_one_immediate_trigger(self):
        """阈值为 1 时立即触发"""
        counter = MotionConsistencyCounter(threshold_frames=1)
        assert counter.update(True) is True
        assert counter.count == 1

    def test_invalid_threshold_raises(self):
        """无效阈值抛出异常"""
        with pytest.raises(ValueError, match="threshold_frames must be >= 1"):
            MotionConsistencyCounter(threshold_frames=0)
        with pytest.raises(ValueError):
            MotionConsistencyCounter(threshold_frames=-1)

    def test_reset_clears_count(self):
        """reset() 清零计数"""
        counter = MotionConsistencyCounter(threshold_frames=5)
        counter.update(True)
        counter.update(True)
        assert counter.count == 2
        counter.reset()
        assert counter.count == 0

    def test_consecutive_false_keeps_at_zero(self):
        """连续 False 保持计数为零"""
        counter = MotionConsistencyCounter(threshold_frames=3)
        counter.update(False)
        counter.update(False)
        counter.update(False)
        assert counter.count == 0

    def test_alternating_true_false_never_reaches_threshold(self):
        """交替 True/False 永远达不到阈值"""
        counter = MotionConsistencyCounter(threshold_frames=3)
        for _ in range(10):
            assert counter.update(True) is False
            assert counter.update(False) is False
        assert counter.count == 0

    def test_after_threshold_stays_true(self):
        """达到阈值后继续 update(True) 仍返回 True"""
        counter = MotionConsistencyCounter(threshold_frames=2)
        counter.update(True)
        counter.update(True)
        assert counter.update(True) is True
        assert counter.update(True) is True
        assert counter.count == 4

    def test_boundary_threshold_n_minus_1(self):
        """N-1 帧时未达到阈值"""
        counter = MotionConsistencyCounter(threshold_frames=5)
        for i in range(4):  # N-1
            assert counter.update(True) is False, f"Failed at frame {i}"
        assert counter.count == 4

    def test_boundary_threshold_n(self):
        """第 N 帧达到阈值"""
        counter = MotionConsistencyCounter(threshold_frames=5)
        for _ in range(4):
            counter.update(True)
        assert counter.update(True) is True
        assert counter.count == 5
