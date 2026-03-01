# tests/flowsentry/test_flow_backend_axelera.py
# Copyright 2025, FlowSentry-Wake

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from flowsentry.config import FlowConfig
from flowsentry.runtime.adapters.flow_backend_axelera import (
    AxeleraFlowBackend,
    _compute_bbox_iou,
    _merge_regions_to_bbox,
    _tensor_to_flow,
)
from flowsentry.types import FlowRegion


class TestTensorToFlow:
    """Test tensor to flow conversion"""

    def test_none_returns_none(self):
        """None input returns None"""
        assert _tensor_to_flow(None) is None

    def test_numpy_array_hw2(self):
        """Numpy array [H, W, 2] is returned as-is"""
        flow = np.random.rand(100, 200, 2).astype(np.float32)
        result = _tensor_to_flow(flow)
        assert result is not None
        assert result.shape == (100, 200, 2)

    def test_numpy_array_2hw(self):
        """Numpy array [2, H, W] is transposed to [H, W, 2]"""
        flow = np.random.rand(2, 100, 200).astype(np.float32)
        result = _tensor_to_flow(flow)
        assert result is not None
        assert result.shape == (100, 200, 2)

    def test_numpy_array_n2hw(self):
        """Numpy array [N, 2, H, W] is converted to [H, W, 2]"""
        flow = np.random.rand(1, 2, 100, 200).astype(np.float32)
        result = _tensor_to_flow(flow)
        assert result is not None
        assert result.shape == (100, 200, 2)

    def test_numpy_array_nhw2(self):
        """Numpy array [N, H, W, 2] is converted to [H, W, 2]"""
        flow = np.random.rand(1, 100, 200, 2).astype(np.float32)
        result = _tensor_to_flow(flow)
        assert result is not None
        assert result.shape == (100, 200, 2)

    def test_numpy_array_nhw4_uses_first_two_channels(self):
        """Numpy array [N, H, W, 4] keeps first 2 channels as flow."""
        flow = np.random.rand(1, 100, 200, 4).astype(np.float32)
        result = _tensor_to_flow(flow)
        assert result is not None
        assert result.shape == (100, 200, 2)

    def test_numpy_array_n4hw_uses_first_two_channels(self):
        """Numpy array [N, 4, H, W] keeps first 2 channels as flow."""
        flow = np.random.rand(1, 4, 100, 200).astype(np.float32)
        result = _tensor_to_flow(flow)
        assert result is not None
        assert result.shape == (100, 200, 2)

    def test_invalid_shape_returns_none(self):
        """Invalid shape returns None"""
        flow = np.random.rand(100, 200)  # 2D
        assert _tensor_to_flow(flow) is None

        flow = np.random.rand(100, 200, 3)  # 3 channels
        assert _tensor_to_flow(flow) is None


class TestComputeBboxIoU:
    """Test bbox IoU computation"""

    def test_none_bboxes(self):
        """None bboxes return 0"""
        assert _compute_bbox_iou(None, (0, 0, 10, 10)) == 0.0
        assert _compute_bbox_iou((0, 0, 10, 10), None) == 0.0
        assert _compute_bbox_iou(None, None) == 0.0

    def test_no_overlap(self):
        """Non-overlapping bboxes return 0"""
        iou = _compute_bbox_iou((0, 0, 10, 10), (20, 20, 30, 30))
        assert iou == 0.0

    def test_full_overlap(self):
        """Same bbox returns 1.0"""
        iou = _compute_bbox_iou((0, 0, 10, 10), (0, 0, 10, 10))
        assert iou == 1.0

    def test_partial_overlap(self):
        """Partial overlap returns correct IoU"""
        iou = _compute_bbox_iou((0, 0, 10, 10), (5, 5, 15, 15))
        assert 0 < iou < 1


class TestMergeRegionsToBbox:
    """Test selecting representative bbox from flow regions"""

    def test_empty_list(self):
        """Empty list returns None"""
        assert _merge_regions_to_bbox([]) is None

    def test_single_region(self):
        """Single region returns its bbox"""
        regions = [FlowRegion(bbox_xyxy=np.array([10, 20, 100, 200]), area=100)]
        bbox = _merge_regions_to_bbox(regions)
        assert bbox == (10.0, 20.0, 100.0, 200.0)

    def test_multiple_regions(self):
        """Multiple regions return the largest region bbox"""
        regions = [
            FlowRegion(bbox_xyxy=np.array([10, 20, 100, 200]), area=100),
            FlowRegion(bbox_xyxy=np.array([50, 60, 150, 250]), area=400),
        ]
        bbox = _merge_regions_to_bbox(regions)
        assert bbox == (50.0, 60.0, 150.0, 250.0)


class _FakeTensor:
    def __init__(self, data: np.ndarray):
        self._data = data

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._data


@dataclass
class _FakeFrameResult:
    tensor: Any = None
    meta: Any = None


class TestAxeleraFlowBackend:
    """Test AxeleraFlowBackend"""

    def test_extract_no_tensor(self):
        """No tensor returns empty output"""
        backend = AxeleraFlowBackend()
        result = backend.extract(_FakeFrameResult(tensor=None))
        assert result.flow_regions == ()
        assert result.flow_bbox is None
        assert result.flow_present is False
        assert result.flow_consistent is False

    def test_extract_zero_flow(self):
        """Zero flow returns no regions"""
        backend = AxeleraFlowBackend()
        flow = np.zeros((100, 200, 2), dtype=np.float32)
        result = backend.extract(_FakeFrameResult(tensor=flow))
        assert result.flow_regions == ()
        assert result.flow_present is False

    def test_extract_with_motion(self):
        """Flow with motion returns regions"""
        config = FlowConfig(
            mask_magnitude_threshold=0.5,
            mask_min_region_area=10,
        )
        backend = AxeleraFlowBackend(config=config)
        flow = np.zeros((100, 200, 2), dtype=np.float32)
        flow[10:30, 10:30, 0] = 2.0
        flow[10:30, 10:30, 1] = 2.0

        result = backend.extract(_FakeFrameResult(tensor=flow))
        assert result.flow_present is True
        assert result.flow_bbox is not None

    def test_consistency_tracking(self):
        """Consistency is tracked across frames"""
        config = FlowConfig(
            mask_magnitude_threshold=0.5,
            mask_min_region_area=10,
        )
        backend = AxeleraFlowBackend(config=config, consistency_iou_threshold=0.3)

        flow1 = np.zeros((100, 200, 2), dtype=np.float32)
        flow1[10:50, 10:50, :] = 2.0
        r1 = backend.extract(_FakeFrameResult(tensor=flow1))
        assert r1.flow_consistent is False

        flow2 = np.zeros((100, 200, 2), dtype=np.float32)
        flow2[12:52, 12:52, :] = 2.0
        r2 = backend.extract(_FakeFrameResult(tensor=flow2))
        assert r2.flow_consistent is True

    def test_reset_clears_state(self):
        """Reset clears previous bbox tracking"""
        backend = AxeleraFlowBackend()
        flow = np.zeros((100, 200, 2), dtype=np.float32)
        flow[10:50, 10:50, :] = 2.0
        backend.extract(_FakeFrameResult(tensor=flow))
        backend.reset()

        flow2 = np.zeros((100, 200, 2), dtype=np.float32)
        flow2[10:50, 10:50, :] = 2.0
        result = backend.extract(_FakeFrameResult(tensor=flow2))
        assert result.flow_consistent is False

    def test_extract_flow_regions(self):
        """extract_flow_regions returns regions only"""
        config = FlowConfig(
            mask_magnitude_threshold=0.5,
            mask_min_region_area=10,
        )
        backend = AxeleraFlowBackend(config=config)
        flow = np.zeros((100, 200, 2), dtype=np.float32)
        flow[10:30, 10:30, :] = 2.0

        regions = backend.extract_flow_regions(_FakeFrameResult(tensor=flow))
        assert len(regions) > 0

    def test_torch_tensor_input(self):
        """PyTorch tensor input works"""
        config = FlowConfig(
            mask_magnitude_threshold=0.5,
            mask_min_region_area=10,
        )
        backend = AxeleraFlowBackend(config=config)
        flow = np.zeros((2, 100, 200), dtype=np.float32)
        flow[:, 10:30, 10:30] = 2.0
        tensor = _FakeTensor(flow)

        result = backend.extract(_FakeFrameResult(tensor=tensor))
        assert result.flow_present is True

    def test_extract_fallback_to_meta_opticalflow(self):
        """Fallback to meta['opticalflow'] when frame_result.tensor is missing"""
        config = FlowConfig(
            mask_magnitude_threshold=0.5,
            mask_min_region_area=10,
        )
        backend = AxeleraFlowBackend(config=config)

        flow = np.zeros((100, 200, 2), dtype=np.float32)
        flow[20:40, 30:60, :] = 2.0
        result = backend.extract(_FakeFrameResult(tensor=None, meta={"opticalflow": flow}))
        assert result.flow_present is True
        assert result.flow_bbox is not None

    def test_extract_fallback_to_meta_task_results(self):
        """Fallback can extract flow from task-like meta.results payload"""
        config = FlowConfig(
            mask_magnitude_threshold=0.5,
            mask_min_region_area=10,
        )
        backend = AxeleraFlowBackend(config=config)

        flow = np.zeros((100, 200, 2), dtype=np.float32)
        flow[10:30, 10:30, :] = 2.0

        class _FakeTaskMeta:
            def __init__(self, results):
                self.results = results

        invalid_tensor = np.zeros((100, 200, 3), dtype=np.float32)
        frame = _FakeFrameResult(
            tensor=invalid_tensor,
            meta={"opticalflow": _FakeTaskMeta([flow])},
        )
        result = backend.extract(frame)
        assert result.flow_present is True
        assert result.flow_bbox is not None

    def test_extract_from_meta_tensor_meta_tensors(self):
        """Can extract flow from TensorMeta-like object exposing tensors list."""
        config = FlowConfig(
            mask_magnitude_threshold=0.5,
            mask_min_region_area=10,
        )
        backend = AxeleraFlowBackend(config=config)

        flow = np.zeros((1, 576, 1024, 4), dtype=np.float32)
        flow[0, 200:240, 300:360, :2] = 2.0

        class _FakeTensorMeta:
            def __init__(self, tensors):
                self.tensors = tensors

        frame = _FakeFrameResult(tensor=None, meta={"opticalflow": _FakeTensorMeta([flow])})
        result = backend.extract(frame)
        assert result.flow_present is True
        assert result.flow_bbox is not None

    def test_extract_does_not_use_flow_image_meta_as_proxy(self):
        """Flow image meta should not be used as a pseudo-flow source."""
        config = FlowConfig(
            mask_magnitude_threshold=0.5,
            mask_min_region_area=10,
        )
        backend = AxeleraFlowBackend(config=config)

        class _FakeFlowImageMeta:
            def __init__(self, img):
                self.img = img

        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[30:60, 40:90, :] = 180
        frame = _FakeFrameResult(tensor=None, meta={"opticalflow": _FakeFlowImageMeta(img)})
        result = backend.extract(frame)
        assert result.flow_present is False
        assert result.flow_bbox is None

    def test_extract_prefers_largest_resolution_tensor_candidate(self):
        """When multiple flow tensors exist, backend should choose largest resolution."""
        config = FlowConfig(
            mask_magnitude_threshold=0.5,
            mask_min_region_area=5,
        )
        backend = AxeleraFlowBackend(config=config)

        low_res = np.zeros((1, 144, 256, 4), dtype=np.float32)
        low_res[0, 10:20, 10:20, :2] = 2.0
        high_res = np.zeros((1, 576, 1024, 4), dtype=np.float32)
        high_res[0, 400:430, 700:740, :2] = 2.0

        frame = _FakeFrameResult(tensor=None, meta={"results": [low_res, high_res]})
        result = backend.extract(frame)

        assert result.flow_present is True
        assert result.flow_bbox is not None
        x1, y1, x2, y2 = result.flow_bbox
        assert x1 <= 700 <= x2
        assert y1 <= 400 <= y2

    def test_extract_emits_debug_info(self):
        """Debug callback receives extraction diagnostics"""
        config = FlowConfig(
            mask_magnitude_threshold=0.5,
            mask_min_region_area=10,
        )
        debug_records: list[dict[str, Any]] = []
        backend = AxeleraFlowBackend(config=config, debug_callback=debug_records.append)

        flow = np.zeros((100, 200, 2), dtype=np.float32)
        flow[5:25, 5:25, :] = 2.0
        result = backend.extract(_FakeFrameResult(tensor=flow))

        assert result.flow_present is True
        assert len(debug_records) == 1
        debug = debug_records[0]
        assert debug["flow_found"] is True
        assert debug["source"] is not None
        assert debug["decode_mode"] == "tensor"
        assert debug["flow_shape"] == (100, 200, 2)
        assert debug["candidate_count"] >= 1
