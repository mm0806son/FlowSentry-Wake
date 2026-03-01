# tests/flowsentry/test_yolo_filter.py
# Copyright 2025, FlowSentry-Wake

import pytest
from flowsentry.types import Detection
from flowsentry.vision.yolo_filter import (
    filter_person_detections,
    person_bboxes_from_detections,
    select_primary_person_bbox,
)


class TestFilterPersonDetections:
    def test_filters_non_person(self):
        """过滤非 person 类别"""
        detections = (
            Detection(bbox_xyxy=(0, 0, 10, 10), class_id=0, class_name="car", confidence=0.9),
            Detection(bbox_xyxy=(0, 0, 20, 20), class_id=1, class_name="person", confidence=0.85),
            Detection(bbox_xyxy=(0, 0, 30, 30), class_id=2, class_name="dog", confidence=0.8),
        )
        result = filter_person_detections(detections)
        assert len(result) == 1
        assert result[0].class_name == "person"

    def test_filters_by_confidence(self):
        """按置信度过滤"""
        detections = (
            Detection(bbox_xyxy=(0, 0, 10, 10), class_id=0, class_name="person", confidence=0.3),
            Detection(bbox_xyxy=(0, 0, 20, 20), class_id=0, class_name="person", confidence=0.7),
        )
        result = filter_person_detections(detections, min_confidence=0.5)
        assert len(result) == 1
        assert result[0].confidence == 0.7

    def test_sorts_by_confidence_desc(self):
        """按置信度降序排序"""
        detections = (
            Detection(bbox_xyxy=(0, 0, 10, 10), class_id=0, class_name="person", confidence=0.5),
            Detection(bbox_xyxy=(0, 0, 20, 20), class_id=0, class_name="person", confidence=0.9),
            Detection(bbox_xyxy=(0, 0, 30, 30), class_id=0, class_name="person", confidence=0.7),
        )
        result = filter_person_detections(detections)
        assert result[0].confidence == 0.9
        assert result[1].confidence == 0.7
        assert result[2].confidence == 0.5

    def test_empty_input(self):
        """空输入返回空"""
        result = filter_person_detections(())
        assert result == ()

    def test_case_insensitive(self):
        """类别名大小写不敏感"""
        detections = (
            Detection(bbox_xyxy=(0, 0, 10, 10), class_id=0, class_name="PERSON", confidence=0.9),
            Detection(bbox_xyxy=(0, 0, 20, 20), class_id=0, class_name="Person", confidence=0.8),
        )
        result = filter_person_detections(detections)
        assert len(result) == 2

    def test_custom_labels(self):
        """自定义标签过滤"""
        detections = (
            Detection(bbox_xyxy=(0, 0, 10, 10), class_id=0, class_name="person", confidence=0.9),
            Detection(bbox_xyxy=(0, 0, 20, 20), class_id=1, class_name="pedestrian", confidence=0.8),
        )
        result = filter_person_detections(detections, person_labels=("person", "pedestrian"))
        assert len(result) == 2


class TestPersonBboxesFromDetections:
    def test_extracts_bboxes(self):
        """提取 bbox"""
        detections = (
            Detection(bbox_xyxy=(10.0, 20.0, 30.0, 40.0), class_id=0, class_name="person", confidence=0.9),
            Detection(bbox_xyxy=(50.0, 60.0, 70.0, 80.0), class_id=0, class_name="person", confidence=0.8),
        )
        bboxes = person_bboxes_from_detections(detections)
        assert len(bboxes) == 2
        assert bboxes[0] == (10.0, 20.0, 30.0, 40.0)
        assert bboxes[1] == (50.0, 60.0, 70.0, 80.0)

    def test_empty_returns_empty(self):
        """空输入返回空"""
        bboxes = person_bboxes_from_detections(())
        assert bboxes == ()


class TestSelectPrimaryPersonBbox:
    def test_selects_largest_area(self):
        """选择面积最大的 bbox"""
        bboxes = (
            (0.0, 0.0, 10.0, 10.0),  # 面积 100
            (0.0, 0.0, 20.0, 20.0),  # 面积 400
            (0.0, 0.0, 5.0, 5.0),    # 面积 25
        )
        result = select_primary_person_bbox(bboxes)
        assert result == (0.0, 0.0, 20.0, 20.0)

    def test_empty_returns_none(self):
        """空输入返回 None"""
        assert select_primary_person_bbox(()) is None

    def test_single_returns_it(self):
        """单个 bbox 直接返回"""
        bbox = (10.0, 20.0, 30.0, 40.0)
        assert select_primary_person_bbox((bbox,)) == bbox
