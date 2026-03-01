# tests/flowsentry/test_frame_diff_monitor.py
# Copyright 2025, FlowSentry-Wake
# 扩展版本

import numpy as np
import pytest

from flowsentry.motion.frame_diff import FrameDiffMonitor, check_frame_diff


# === 原有测试（保持不变）===

def test_frame_diff_monitor_exists_and_has_update():
    """FrameDiffMonitor 存在且具备 update 方法"""
    mon = FrameDiffMonitor()
    assert hasattr(mon, "update")
    assert callable(mon.update)


def test_frame_diff_monitor_update_raises_until_implemented():
    """当前占位实现下 update 抛出 NotImplementedError"""
    mon = FrameDiffMonitor()
    with pytest.raises(NotImplementedError, match="FrameDiffMonitor"):
        mon.update(None)


# === 新增测试 ===

class TestCheckFrameDiff:
    def test_identical_frames_no_trigger(self):
        """相同帧不触发"""
        frame = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = check_frame_diff(
            frame, frame,
            min_area=10,
            pixel_change_ratio_threshold=0.01,
        )
        assert result.triggered is False
        assert result.change_ratio == 0.0
        assert result.region_area == 0

    def test_small_change_below_threshold_no_trigger(self):
        """变化小于阈值不触发"""
        prev = np.zeros((100, 100), dtype=np.uint8)
        curr = np.zeros((100, 100), dtype=np.uint8)
        curr[0:5, 0:5] = 255  # 25 像素变化，占 0.25%
        result = check_frame_diff(
            prev, curr,
            min_area=10,
            pixel_change_ratio_threshold=0.01,  # 1%
        )
        assert result.triggered is False
        assert result.change_ratio < 0.01

    def test_large_change_above_threshold_triggers(self):
        """变化超过阈值且区域足够大时触发"""
        prev = np.zeros((100, 100), dtype=np.uint8)
        curr = np.zeros((100, 100), dtype=np.uint8)
        curr[0:30, 0:30] = 255  # 900 像素变化，占 9%
        result = check_frame_diff(
            prev, curr,
            min_area=100,
            pixel_change_ratio_threshold=0.01,
        )
        assert result.triggered is True
        assert result.change_ratio >= 0.01
        assert result.region_area >= 100
        assert result.bbox_xyxy is not None

    def test_change_above_ratio_but_small_area_no_trigger(self):
        """变化比例够但区域小于 min_area 时不触发"""
        prev = np.zeros((10, 10), dtype=np.uint8)
        curr = np.zeros((10, 10), dtype=np.uint8)
        curr[0:3, 0:3] = 255  # 9 像素，占 9%
        result = check_frame_diff(
            prev, curr,
            min_area=100,  # 要求至少 100 像素
            pixel_change_ratio_threshold=0.01,
        )
        assert result.triggered is False

    def test_bbox_coordinates_correct(self):
        """返回的 bbox 坐标正确"""
        prev = np.zeros((64, 64), dtype=np.uint8)
        curr = np.zeros((64, 64), dtype=np.uint8)
        curr[10:30, 15:40] = 255  # y:10-29, x:15-39
        result = check_frame_diff(
            prev, curr,
            min_area=10,
            pixel_change_ratio_threshold=0.001,
        )
        assert result.triggered is True
        x1, y1, x2, y2 = result.bbox_xyxy
        assert x1 <= 15 and x2 >= 39
        assert y1 <= 10 and y2 >= 29

    def test_color_frames_converted_to_gray(self):
        """彩色帧自动转灰度"""
        prev = np.zeros((50, 50, 3), dtype=np.uint8)
        curr = np.zeros((50, 50, 3), dtype=np.uint8)
        curr[10:20, 10:20] = [255, 0, 0]
        result = check_frame_diff(
            prev, curr,
            min_area=10,
            pixel_change_ratio_threshold=0.001,
        )
        assert result.triggered is True

    def test_shape_mismatch_raises(self):
        """帧形状不匹配抛出异常"""
        prev = np.zeros((50, 50), dtype=np.uint8)
        curr = np.zeros((60, 60), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape mismatch"):
            check_frame_diff(prev, curr, min_area=10, pixel_change_ratio_threshold=0.01)


class TestFrameDiffMonitorCheck:
    def test_check_method_delegates(self):
        """check() 方法正确委托给 check_frame_diff()"""
        mon = FrameDiffMonitor(min_area=50, pixel_change_ratio_threshold=0.02)
        prev = np.zeros((100, 100), dtype=np.uint8)
        curr = np.zeros((100, 100), dtype=np.uint8)
        curr[0:40, 0:40] = 255
        result = mon.check(prev, curr)
        assert result.triggered is True
