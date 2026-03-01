# tests/flowsentry/test_iou_matcher.py
# Copyright 2025, FlowSentry-Wake
# 扩展版本

import pytest
from flowsentry.fusion.iou_matcher import IoUMatcher, IoUMatchResult


# === 原有测试（保持不变）===

def test_iou_matcher_returns_best_person_bbox_and_match_flag():
    matcher = IoUMatcher(iou_threshold=0.3)
    flow_bbox = (10.0, 10.0, 30.0, 30.0)
    persons = (
        (0.0, 0.0, 5.0, 5.0),
        (12.0, 12.0, 28.0, 28.0),
    )

    result = matcher.match(flow_bbox, persons)

    assert result.matched is True
    assert result.best_person_bbox == persons[1]
    assert result.best_iou is not None
    assert result.best_iou > 0.5


def test_iou_matcher_handles_missing_inputs():
    matcher = IoUMatcher(iou_threshold=0.5)

    no_flow = matcher.match(None, ((1.0, 1.0, 2.0, 2.0),))
    assert no_flow.matched is False
    assert no_flow.best_iou is None
    assert no_flow.best_person_bbox is None

    no_person = matcher.match((1.0, 1.0, 2.0, 2.0), ())
    assert no_person.matched is False
    assert no_person.best_iou is None


# === 新增测试 ===

class TestIoUMatcher:
    def test_invalid_threshold_raises(self):
        """无效阈值抛出异常"""
        with pytest.raises(ValueError, match="iou_threshold must be in"):
            IoUMatcher(iou_threshold=-0.1)
        with pytest.raises(ValueError):
            IoUMatcher(iou_threshold=1.1)

    def test_exact_threshold_boundary(self):
        """IoU 恰好等于阈值时匹配"""
        matcher = IoUMatcher(iou_threshold=0.5)
        flow_bbox = (0.0, 0.0, 10.0, 10.0)
        person_bbox = (5.0, 0.0, 15.0, 10.0)
        result = matcher.match(flow_bbox, (person_bbox,))
        assert result.best_iou is not None

    def test_no_overlap_zero_iou(self):
        """无交集时 IoU 为 0"""
        matcher = IoUMatcher(iou_threshold=0.3)
        flow_bbox = (0.0, 0.0, 10.0, 10.0)
        person_bbox = (100.0, 100.0, 110.0, 110.0)
        result = matcher.match(flow_bbox, (person_bbox,))
        assert result.best_iou == 0.0
        assert result.matched is False

    def test_perfect_overlap_iou_one(self):
        """完全重叠时 IoU 为 1"""
        matcher = IoUMatcher(iou_threshold=0.3)
        bbox = (10.0, 20.0, 30.0, 40.0)
        result = matcher.match(bbox, (bbox,))
        assert result.best_iou == 1.0
        assert result.matched is True

    def test_selects_best_iou_from_multiple(self):
        """多个 person bbox 时选择最高 IoU"""
        matcher = IoUMatcher(iou_threshold=0.3)
        flow_bbox = (10.0, 10.0, 50.0, 50.0)
        persons = (
            (100.0, 100.0, 110.0, 110.0),  # 无交集
            (15.0, 15.0, 45.0, 45.0),      # 高 IoU
            (12.0, 12.0, 20.0, 20.0),      # 低 IoU
        )
        result = matcher.match(flow_bbox, persons)
        assert result.best_person_bbox == persons[1]
        assert result.best_iou is not None
        assert result.best_iou > 0.5

    def test_result_contains_threshold(self):
        """结果包含阈值"""
        matcher = IoUMatcher(iou_threshold=0.4)
        result = matcher.match((0, 0, 10, 10), ((0, 0, 10, 10),))
        assert result.threshold == 0.4


class TestIoUMatchResult:
    def test_frozen_dataclass(self):
        """结果是不可变的"""
        result = IoUMatchResult(matched=True, best_iou=0.8, best_person_bbox=(0, 0, 10, 10), threshold=0.3)
        with pytest.raises(Exception):
            result.matched = False
