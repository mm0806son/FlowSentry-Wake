# tests/flowsentry/conftest.py
# Copyright 2025, FlowSentry-Wake
# 共用 fixture：BBox、person_bboxes、配置、模拟帧等

import numpy as np
import pytest

from flowsentry.types import Detection, FrameSignals
from flowsentry.config import TriageConfig


# === BBox fixtures ===

@pytest.fixture
def flow_bbox_xyxy():
    """单个光流区域 bbox (x1,y1,x2,y2)"""
    return (10.0, 20.0, 100.0, 120.0)


@pytest.fixture
def person_bbox_high_iou():
    """与 flow_bbox 高 IoU 的 person 框"""
    return (15.0, 25.0, 95.0, 115.0)


@pytest.fixture
def person_bbox_no_overlap():
    """与 flow_bbox 无交集的 person 框"""
    return (200.0, 200.0, 300.0, 300.0)


@pytest.fixture
def person_bboxes_empty():
    return ()


@pytest.fixture
def person_bboxes_multiple():
    """多个 person bbox"""
    return (
        (10.0, 10.0, 50.0, 50.0),
        (60.0, 60.0, 100.0, 100.0),
        (110.0, 110.0, 150.0, 150.0),
    )


# === Frame fixtures ===

@pytest.fixture
def blank_frame_gray():
    """空白灰度帧"""
    return np.zeros((100, 100), dtype=np.uint8)


@pytest.fixture
def frame_with_motion():
    """有运动区域的帧"""
    frame = np.zeros((100, 100), dtype=np.uint8)
    frame[20:50, 30:60] = 255
    return frame


@pytest.fixture
def frame_pair_with_motion():
    """有运动差异的帧对"""
    prev = np.zeros((100, 100), dtype=np.uint8)
    curr = np.zeros((100, 100), dtype=np.uint8)
    curr[20:50, 30:60] = 255
    return prev, curr


# === Flow fixtures ===

@pytest.fixture
def zero_flow():
    """零光流场"""
    return np.zeros((64, 64, 2), dtype=np.float32)


@pytest.fixture
def flow_with_motion():
    """有运动的光流场"""
    flow = np.zeros((64, 64, 2), dtype=np.float32)
    flow[10:30, 15:40, 0] = 5.0  # x 方向运动
    flow[10:30, 15:40, 1] = 3.0  # y 方向运动
    return flow


# === Detection fixtures ===

@pytest.fixture
def person_detection():
    """单个 person 检测"""
    return Detection(
        bbox_xyxy=(10.0, 20.0, 100.0, 200.0),
        class_id=0,
        class_name="person",
        confidence=0.95,
    )


@pytest.fixture
def mixed_detections():
    """混合类别检测"""
    return (
        Detection(bbox_xyxy=(0, 0, 10, 10), class_id=0, class_name="person", confidence=0.9),
        Detection(bbox_xyxy=(0, 0, 20, 20), class_id=1, class_name="car", confidence=0.85),
        Detection(bbox_xyxy=(0, 0, 30, 30), class_id=2, class_name="dog", confidence=0.8),
    )


# === Config fixtures ===

@pytest.fixture
def default_config():
    """默认配置"""
    return TriageConfig()


@pytest.fixture
def fast_trigger_config():
    """快速触发配置（用于测试）"""
    cfg = TriageConfig()
    cfg.flow.consistency_frames_threshold = 1
    cfg.runtime.no_motion_reset_frames = 1
    return cfg


# === FrameSignals fixtures ===

@pytest.fixture
def idle_signals():
    """空闲状态信号"""
    return FrameSignals(
        frame_diff_triggered=False,
        flow_present=False,
        flow_consistent=False,
        flow_bbox=None,
        person_bboxes=(),
    )


@pytest.fixture
def triggered_signals():
    """触发状态信号"""
    return FrameSignals(
        frame_diff_triggered=True,
        flow_present=True,
        flow_consistent=True,
        flow_bbox=(10.0, 10.0, 50.0, 50.0),
        person_bboxes=(),
    )
