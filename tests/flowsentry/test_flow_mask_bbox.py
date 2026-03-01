# tests/flowsentry/test_flow_mask_bbox.py
# Copyright 2025, FlowSentry-Wake
# 扩展版本

import numpy as np
import pytest

from flowsentry.motion.flow_mask import bbox_from_binary_mask, flow_to_mask_bboxes


# === 原有测试（保持不变）===

def test_bbox_from_binary_mask_empty_returns_none():
    """全零掩码返回 None"""
    mask = np.zeros((64, 64), dtype=np.uint8)
    assert bbox_from_binary_mask(mask, min_area=1) is None


def test_bbox_from_binary_mask_single_region_returns_bbox():
    """一块运动区域返回 (x1,y1,x2,y2)"""
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 15:40] = 255
    bbox = bbox_from_binary_mask(mask, min_area=10)
    assert bbox is not None
    assert len(bbox) == 4
    assert bbox[0] <= bbox[2] and bbox[1] <= bbox[3]
    assert bbox == (15.0, 10.0, 39.0, 29.0)


def test_bbox_from_binary_mask_small_region_filtered():
    """小于 min_region_area 时返回 None"""
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20, 20] = 255
    bbox = bbox_from_binary_mask(mask, min_area=100)
    assert bbox is None


def test_bbox_from_binary_mask_requires_2d():
    """非 2D 掩码抛出 ValueError"""
    with pytest.raises(ValueError, match="Expected 2D mask"):
        bbox_from_binary_mask(np.zeros((2, 3, 4)))


# === 新增测试 ===

class TestBboxFromBinaryMask:
    def test_multiple_regions_returns_largest_bbox(self):
        """多个区域返回包围所有区域的 bbox"""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        mask[50:60, 70:80] = 255
        bbox = bbox_from_binary_mask(mask, min_area=1)
        assert bbox is not None
        x1, y1, x2, y2 = bbox
        assert x1 == 10 and x2 == 79
        assert y1 == 10 and y2 == 59

    def test_min_area_exactly_at_boundary(self):
        """min_area 边界值测试"""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:3, 0:3] = 255  # 9 像素
        assert bbox_from_binary_mask(mask, min_area=9) is not None
        assert bbox_from_binary_mask(mask, min_area=10) is None


class TestFlowToMaskBboxes:
    def test_zero_flow_returns_empty(self):
        """零光流返回空列表"""
        flow = np.zeros((64, 64, 2), dtype=np.float32)
        regions = flow_to_mask_bboxes(
            flow,
            magnitude_threshold=1.0,
            min_region_area=10,
        )
        assert regions == []

    def test_single_motion_region(self):
        """单一运动区域"""
        flow = np.zeros((64, 64, 2), dtype=np.float32)
        flow[10:30, 15:40, 0] = 5.0  # x 方向运动
        regions = flow_to_mask_bboxes(
            flow,
            magnitude_threshold=1.0,
            min_region_area=10,
        )
        assert len(regions) >= 1
        assert all(r.area >= 10 for r in regions)

    def test_magnitude_threshold_filters_weak_motion(self):
        """magnitude_threshold 过滤弱运动"""
        flow = np.zeros((64, 64, 2), dtype=np.float32)
        flow[10:20, 10:20, 0] = 0.5  # 低于阈值
        regions = flow_to_mask_bboxes(
            flow,
            magnitude_threshold=1.0,
            min_region_area=1,
        )
        assert regions == []

    def test_min_region_area_filters_small_regions(self):
        """min_region_area 过滤小区域"""
        flow = np.zeros((64, 64, 2), dtype=np.float32)
        flow[10:12, 10:12, 0] = 5.0  # 4 像素
        regions = flow_to_mask_bboxes(
            flow,
            magnitude_threshold=1.0,
            min_region_area=100,
        )
        assert regions == []

    def test_diagonal_motion(self):
        """对角运动（x 和 y 都有分量）"""
        flow = np.zeros((64, 64, 2), dtype=np.float32)
        flow[20:40, 20:40, 0] = 3.0
        flow[20:40, 20:40, 1] = 4.0  # magnitude = 5
        regions = flow_to_mask_bboxes(
            flow,
            magnitude_threshold=4.0,  # 低于 sqrt(9+16)=5
            min_region_area=10,
        )
        assert len(regions) >= 1

    def test_invalid_flow_shape_raises(self):
        """无效光流形状抛出异常"""
        flow = np.zeros((64, 64, 3), dtype=np.float32)  # 错误的通道数
        with pytest.raises(ValueError, match="Expected flow shape"):
            flow_to_mask_bboxes(flow, magnitude_threshold=1.0, min_region_area=1)

    def test_flow_region_has_correct_bbox(self):
        """FlowRegion 的 bbox 坐标正确"""
        flow = np.zeros((100, 100, 2), dtype=np.float32)
        flow[10:30, 20:50, 0] = 5.0
        regions = flow_to_mask_bboxes(
            flow,
            magnitude_threshold=1.0,
            min_region_area=10,
        )
        assert len(regions) >= 1
        region = regions[0]
        x1, y1, x2, y2 = region.bbox_xyxy
        assert x1 <= 20 and x2 >= 49
        assert y1 <= 10 and y2 >= 29
