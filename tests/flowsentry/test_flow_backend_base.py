# tests/flowsentry/test_flow_backend_base.py
# Copyright 2025, FlowSentry-Wake

import pytest
import numpy as np

from flowsentry.runtime.adapters.flow_backend_base import FlowBackend, FlowBackendOutput
from flowsentry.runtime.adapters.flow_backend_mock import MockFlowBackend
from flowsentry.types import FlowRegion


class TestFlowBackendProtocol:
    """Test FlowBackend protocol"""

    def test_protocol_has_required_methods(self):
        """FlowBackend protocol has required methods"""
        assert hasattr(FlowBackend, "extract")
        assert hasattr(FlowBackend, "extract_flow_regions")
        assert hasattr(FlowBackend, "reset")


class TestFlowBackendOutput:
    def test_output_creation_with_flow(self):
        """Create output with flow present"""
        regions = (FlowRegion(bbox_xyxy=np.array([0, 0, 10, 10]), area=100),)
        output = FlowBackendOutput(
            flow_regions=regions,
            flow_bbox=(0.0, 0.0, 10.0, 10.0),
            flow_present=True,
            flow_consistent=True,
        )
        assert len(output.flow_regions) == 1
        assert output.flow_bbox == (0.0, 0.0, 10.0, 10.0)
        assert output.flow_present is True
        assert output.flow_consistent is True

    def test_output_creation_no_flow(self):
        """Create output with no flow"""
        output = FlowBackendOutput(
            flow_regions=(),
            flow_bbox=None,
            flow_present=False,
            flow_consistent=False,
        )
        assert len(output.flow_regions) == 0
        assert output.flow_bbox is None
        assert output.flow_present is False

    def test_output_frozen(self):
        """Output is immutable"""
        output = FlowBackendOutput(
            flow_regions=(),
            flow_bbox=None,
            flow_present=False,
            flow_consistent=False,
        )
        with pytest.raises(Exception):
            output.flow_present = True


class TestMockFlowBackend:
    def test_extract_returns_empty_by_default(self):
        """Returns empty output by default"""
        backend = MockFlowBackend()
        result = backend.extract(None)
        assert result.flow_regions == ()
        assert result.flow_bbox is None
        assert result.flow_present is False
        assert result.flow_consistent is False

    def test_extract_flow_regions_returns_tuple(self):
        """extract_flow_regions returns tuple"""
        backend = MockFlowBackend()
        result = backend.extract_flow_regions(None)
        assert isinstance(result, tuple)

    def test_push_flow_regions(self):
        """Push and extract flow regions"""
        backend = MockFlowBackend()
        regions = (FlowRegion(bbox_xyxy=np.array([10, 20, 100, 200]), area=90 * 180),)
        backend.push_flow_regions(regions)
        result = backend.extract(None)
        assert len(result.flow_regions) == 1
        assert result.flow_present is True

    def test_push_flow_bbox(self):
        """Push flow bbox directly"""
        backend = MockFlowBackend()
        backend.push_flow_bbox((10.0, 20.0, 100.0, 200.0))
        result = backend.extract(None)
        assert result.flow_bbox == (10.0, 20.0, 100.0, 200.0)
        assert result.flow_present is True

    def test_init_with_batches(self):
        """Initialize with batches"""
        backend = MockFlowBackend(batches=[
            (True, True, (10.0, 10.0, 50.0, 50.0)),
            (False, False, None),
            (True, True, (20.0, 20.0, 60.0, 60.0)),
        ])
        r1 = backend.extract(None)
        assert r1.flow_present is True
        assert r1.flow_consistent is True
        r2 = backend.extract(None)
        assert r2.flow_present is False
        r3 = backend.extract(None)
        assert r3.flow_present is True

    def test_reset_clears_state(self):
        """Reset clears internal state"""
        backend = MockFlowBackend()
        backend.push_flow_bbox((10, 10, 50, 50))
        backend.reset()
        result = backend.extract(None)
        assert result.flow_present is False

    def test_queue_fifo_order(self):
        """Queue FIFO order"""
        backend = MockFlowBackend()
        backend.push_flow_bbox((1, 1, 2, 2))
        backend.push_flow_bbox((3, 3, 4, 4))
        r1 = backend.extract(None)
        r2 = backend.extract(None)
        assert r1.flow_bbox == (1, 1, 2, 2)
        assert r2.flow_bbox == (3, 3, 4, 4)

    def test_flow_consistent_parameter(self):
        """Flow consistent can be set independently"""
        backend = MockFlowBackend()
        backend.push_flow(present=True, consistent=False)
        result = backend.extract(None)
        assert result.flow_present is True
        assert result.flow_consistent is False
