# tests/flowsentry/test_yolo_backend_base.py
# Copyright 2025, FlowSentry-Wake

import pytest
from flowsentry.runtime.adapters.yolo_backend_base import YoloBackend, YoloBackendOutput
from flowsentry.runtime.adapters.yolo_backend_mock import MockYoloBackend
from flowsentry.types import BBox, Detection


class TestYoloBackendProtocol:
    """测试 YoloBackend 协议"""

    def test_protocol_has_required_methods(self):
        """YoloBackend 协议有必需的方法"""
        assert hasattr(YoloBackend, "extract")
        assert hasattr(YoloBackend, "extract_person_bboxes")


class TestYoloBackendOutput:
    def test_output_creation(self):
        """创建输出"""
        output = YoloBackendOutput(
            detections=(Detection(bbox_xyxy=(0, 0, 10, 10), class_id=0, class_name="person", confidence=0.9),),
            person_bboxes=((0.0, 0.0, 10.0, 10.0),),
        )
        assert len(output.detections) == 1
        assert len(output.person_bboxes) == 1

    def test_output_frozen(self):
        """输出是不可变的"""
        output = YoloBackendOutput(detections=(), person_bboxes=())
        with pytest.raises(Exception):
            output.detections = ()


class TestMockYoloBackend:
    def test_extract_returns_empty_when_queue_empty(self):
        """队列为空时返回空"""
        backend = MockYoloBackend()
        result = backend.extract(None)
        assert result.detections == ()
        assert result.person_bboxes == ()

    def test_extract_person_bboxes_returns_tuple(self):
        """extract_person_bboxes 返回 tuple"""
        backend = MockYoloBackend()
        result = backend.extract_person_bboxes(None)
        assert isinstance(result, tuple)

    def test_push_and_extract(self):
        """推送和提取"""
        backend = MockYoloBackend()
        backend.push_person_bboxes(((10.0, 20.0, 100.0, 200.0),))
        result = backend.extract(None)
        assert len(result.person_bboxes) == 1
        assert result.person_bboxes[0] == (10.0, 20.0, 100.0, 200.0)

    def test_init_with_batches(self):
        """使用 batches 初始化"""
        backend = MockYoloBackend(batches=[
            ((10.0, 10.0, 50.0, 50.0),),
            ((20.0, 20.0, 60.0, 60.0), (70.0, 70.0, 100.0, 100.0)),
        ])
        r1 = backend.extract(None)
        assert len(r1.person_bboxes) == 1
        r2 = backend.extract(None)
        assert len(r2.person_bboxes) == 2

    def test_creates_detection_for_each_bbox(self):
        """为每个 bbox 创建 Detection"""
        backend = MockYoloBackend()
        backend.push_person_bboxes(((0, 0, 10, 10), (20, 20, 30, 30)))
        result = backend.extract(None)
        assert len(result.detections) == 2
        for det in result.detections:
            assert det.class_name == "person"
            assert det.confidence == 1.0

    def test_queue_fifo_order(self):
        """队列 FIFO 顺序"""
        backend = MockYoloBackend()
        backend.push_person_bboxes(((1, 1, 2, 2),))
        backend.push_person_bboxes(((3, 3, 4, 4),))
        r1 = backend.extract_person_bboxes(None)
        r2 = backend.extract_person_bboxes(None)
        assert r1[0] == (1, 1, 2, 2)
        assert r2[0] == (3, 3, 4, 4)
