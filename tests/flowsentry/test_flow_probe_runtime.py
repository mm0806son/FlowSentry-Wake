# tests/flowsentry/test_flow_probe_runtime.py
# Copyright 2025, FlowSentry-Wake
# Runtime tests for flow probe - requires Axelera environment

from __future__ import annotations

from dataclasses import dataclass
import os
import time
import pytest

from flowsentry.config import FlowConfig
from flowsentry.runtime.adapters import AxeleraFlowBackend, MockFlowBackend
from flowsentry.runtime.flow_probe import (
    FlowProbePreflight,
    FlowProbeBBoxMeta,
    _attach_flow_bbox_overlay_meta,
    validate_axelera_runtime_env,
    list_flow_network_candidates,
    run_axelera_flow_probe,
    run_axelera_flow_probe_inference_core,
)


requires_axelera_runtime = pytest.mark.skipif(
    not os.environ.get("AXELERA_FRAMEWORK"),
    reason="Requires AXELERA runtime environment (AXELERA_FRAMEWORK not set)",
)


class TestValidateAxeleraRuntimeEnv:
    def test_missing_framework_env(self):
        """Missing AXELERA_FRAMEWORK raises error"""
        with pytest.raises(ValueError, match="missing AXELERA_FRAMEWORK"):
            validate_axelera_runtime_env({})

    def test_missing_ld_library_path(self):
        """Missing runtime in LD_LIBRARY_PATH raises error"""
        with pytest.raises(ValueError, match="incomplete"):
            validate_axelera_runtime_env({
                "AXELERA_FRAMEWORK": "/opt/axelera",
                "LD_LIBRARY_PATH": "/usr/lib",
            })

    def test_valid_env(self):
        """Valid environment passes"""
        validate_axelera_runtime_env({
            "AXELERA_FRAMEWORK": "/opt/axelera",
            "LD_LIBRARY_PATH": "/opt/axelera/runtime/lib",
        })


class TestAxeleraFlowBackendWithConfig:
    def test_backend_with_custom_config(self):
        """Backend with custom config"""
        config = FlowConfig(
            mask_magnitude_threshold=2.0,
            mask_min_region_area=500,
        )
        backend = AxeleraFlowBackend(config=config, consistency_iou_threshold=0.5)
        assert backend.config.mask_magnitude_threshold == 2.0
        assert backend.config.mask_min_region_area == 500
        assert backend.consistency_iou_threshold == 0.5

    def test_backend_reset(self):
        """Backend reset clears state"""
        backend = AxeleraFlowBackend()
        backend._prev_bbox = (10, 10, 50, 50)
        backend.reset()
        assert backend._prev_bbox is None


@requires_axelera_runtime
class TestFlowProbeRuntime:
    """Runtime tests that require Axelera environment"""

    def test_list_flow_network_candidates(self):
        """List flow network candidates from registry"""
        candidates = list_flow_network_candidates()
        assert isinstance(candidates, list)
        for name in candidates:
            assert isinstance(name, str)


class _FakeFlowImage:
    def __init__(self, data):
        self._data = data

    def asarray(self):
        return self._data


@dataclass
class _FakeFlowTensor:
    data: object

    def numpy(self):
        return self.data


@dataclass
class _FakeFlowFrameResult:
    tensor: _FakeFlowTensor
    image: _FakeFlowImage | None
    stream_id: int = 0
    src_timestamp: float | None = None


class _FakeFlowStream:
    def __init__(self, frames):
        self._frames = list(frames)
        self.stopped = False

    def __iter__(self):
        return iter(self._frames)

    def stop(self):
        self.stopped = True


@dataclass
class _FakeFlowEvent:
    result: _FakeFlowFrameResult | None


class _FakeFlowEventStream(_FakeFlowStream):
    def with_events(self):
        for frame in self._frames:
            yield _FakeFlowEvent(result=frame)


class TestFlowProbeWithOverlay:
    def test_run_axelera_flow_probe_inference_core_extracts_records(self):
        backend = MockFlowBackend(batches=[(True, True, (1, 2, 3, 4)), (False, False, None)])
        now = time.time()
        fake_stream = _FakeFlowEventStream([
            _FakeFlowFrameResult(
                _FakeFlowTensor(None),
                None,
                stream_id=7,
                src_timestamp=now - 0.2,
            ),
            _FakeFlowFrameResult(
                _FakeFlowTensor(None),
                None,
                stream_id=7,
                src_timestamp=now - 0.1,
            ),
        ])

        def _factory(**kwargs):
            return fake_stream

        records, latency = run_axelera_flow_probe_inference_core(
            network="edgeflownet-opticalflow",
            source="rtsp://example",
            max_frames=2,
            flow_backend=backend,
            stream_factory=_factory,
            display_enabled=False,
        )

        assert fake_stream.stopped is True
        assert len(records) == 2
        assert records[0].flow_present is True
        assert records[1].flow_present is False
        assert latency.sampled_frames == 2
        assert latency.max_ms is not None
        assert latency.max_ms > 0

    def test_run_axelera_flow_probe_inference_core_can_skip_extract(self):
        now = time.time()
        fake_stream = _FakeFlowEventStream([
            _FakeFlowFrameResult(
                _FakeFlowTensor(None),
                None,
                stream_id=9,
                src_timestamp=now - 0.1,
            ),
        ])

        class _NoExtractBackend:
            def extract(self, frame_result):
                raise AssertionError("extract should not be called when flow extract is disabled")

            def extract_flow_regions(self, frame_result):
                return ()

            def reset(self):
                return None

        def _factory(**kwargs):
            return fake_stream

        records, latency = run_axelera_flow_probe_inference_core(
            network="edgeflownet-opticalflow",
            source="rtsp://example",
            max_frames=1,
            flow_backend=_NoExtractBackend(),
            stream_factory=_factory,
            display_enabled=False,
            enable_flow_extract=False,
        )

        assert fake_stream.stopped is True
        assert len(records) == 1
        assert records[0].flow_present is False
        assert latency.sampled_frames == 1

    def test_run_axelera_flow_probe_uses_inference_like_stream_builder(self, monkeypatch):
        from flowsentry.runtime import flow_probe as flow_probe_module

        backend = MockFlowBackend(batches=[(False, False, None)])
        fake_stream = _FakeFlowStream([
            _FakeFlowFrameResult(
                _FakeFlowTensor(None),
                None,
                stream_id=3,
            ),
        ])
        called = {}

        def _fake_preflight(**kwargs):
            return FlowProbePreflight(
                requested_network=kwargs["network"],
                resolved_network="/tmp/resolved-edgeflownet.yaml",
                source=kwargs["source"],
                pipe_type=kwargs["pipe_type"],
            )

        def _fake_stream_builder(*, network, source, pipe_type, stream_kwargs):
            called["network"] = network
            called["source"] = source
            called["pipe_type"] = pipe_type
            called["stream_kwargs"] = dict(stream_kwargs)
            return fake_stream

        monkeypatch.setattr(flow_probe_module, "preflight_axelera_flow_probe", _fake_preflight)
        monkeypatch.setattr(
            flow_probe_module,
            "_create_inference_stream_like_inference",
            _fake_stream_builder,
        )

        records = run_axelera_flow_probe(
            network="edgeflownet-opticalflow",
            source="rtsp://example",
            max_frames=1,
            flow_backend=backend,
            stream_kwargs={"rtsp_latency": 120},
        )

        assert called["network"] == "/tmp/resolved-edgeflownet.yaml"
        assert called["source"] == "rtsp://example"
        assert called["pipe_type"] == "gst"
        assert called["stream_kwargs"] == {"rtsp_latency": 120}
        assert fake_stream.stopped is True
        assert len(records) == 1

    def test_run_axelera_flow_probe_with_overlay_extracts_records(self):
        import numpy as np
        from flowsentry.runtime.flow_probe import run_axelera_flow_probe_with_overlay

        backend = AxeleraFlowBackend(config=FlowConfig())
        fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
        fake_flow = np.zeros((100, 100, 2), dtype=np.float32)

        fake_stream = _FakeFlowStream([
            _FakeFlowFrameResult(
                _FakeFlowTensor(fake_flow),
                _FakeFlowImage(fake_image),
                stream_id=0,
            ),
        ])

        def _factory(**kwargs):
            return fake_stream

        records = run_axelera_flow_probe_with_overlay(
            network="edgeflownet-opticalflow",
            source="fakevideo",
            max_frames=1,
            flow_backend=backend,
            stream_factory=_factory,
        )

        assert len(records) == 1
        assert records[0].flow_present is False
        assert fake_stream.stopped is True


class TestNativeDisplayOverlayMeta:
    def test_attach_flow_bbox_overlay_meta_into_existing_axmeta(self):
        from axelera.app.meta.base import AxMeta

        meta = AxMeta(image_id="flow-probe-test")
        out = _attach_flow_bbox_overlay_meta(
            meta,
            (1.0, 2.0, 30.0, 40.0),
            frame_index=1,
            stream_id=0,
        )

        assert out is meta
        assert "flowsentry_flow_bbox" in list(meta)
        flow_meta = meta["flowsentry_flow_bbox"]
        assert isinstance(flow_meta, FlowProbeBBoxMeta)
        assert flow_meta.flow_bbox == (1.0, 2.0, 30.0, 40.0)

    def test_attach_flow_bbox_overlay_meta_creates_axmeta_when_missing(self):
        out = _attach_flow_bbox_overlay_meta(
            None,
            (5.0, 6.0, 70.0, 80.0),
            frame_index=2,
            stream_id=3,
        )

        assert out is not None
        assert "flowsentry_flow_bbox" in list(out)

    def test_non_native_display_converts_rgb_to_bgr_and_draws_bbox(self, monkeypatch):
        import numpy as np
        import cv2
        from flowsentry.overlay import OverlayConfig, OverlayRenderer
        from flowsentry.runtime.flow_probe import run_axelera_flow_probe_with_overlay
        from flowsentry.runtime.adapters import MockFlowBackend

        captured: dict[str, np.ndarray] = {}

        def _fake_imshow(_title, frame):
            captured["frame"] = frame.copy()

        monkeypatch.setattr(cv2, "imshow", _fake_imshow)
        monkeypatch.setattr(cv2, "waitKey", lambda _delay: -1)

        backend = MockFlowBackend(batches=[(True, False, (10.0, 10.0, 40.0, 40.0))])
        fake_image = np.zeros((80, 80, 3), dtype=np.uint8)
        fake_image[:, :, 0] = 255  # RGB red
        fake_flow = np.zeros((80, 80, 2), dtype=np.float32)

        fake_stream = _FakeFlowStream([
            _FakeFlowFrameResult(
                _FakeFlowTensor(fake_flow),
                _FakeFlowImage(fake_image),
                stream_id=0,
            ),
        ])

        def _factory(**kwargs):
            return fake_stream

        renderer = OverlayRenderer(OverlayConfig())
        records = run_axelera_flow_probe_with_overlay(
            network="edgeflownet-opticalflow",
            source="fakevideo",
            max_frames=1,
            flow_backend=backend,
            stream_factory=_factory,
            overlay_renderer=renderer,
            display=True,
            use_native_display=False,
        )

        assert len(records) == 1
        assert records[0].flow_bbox == (10.0, 10.0, 40.0, 40.0)
        assert records[0].annotated_frame is not None
        assert "frame" in captured
        # imshow receives BGR; RGB red should become BGR red=(0,0,255)
        assert tuple(captured["frame"][0, 0].tolist()) == (0, 0, 255)

    def test_run_axelera_flow_probe_with_overlay_renders_frames(self):
        import numpy as np
        from flowsentry.overlay import OverlayConfig, OverlayRenderer
        from flowsentry.runtime.flow_probe import run_axelera_flow_probe_with_overlay

        backend = AxeleraFlowBackend(config=FlowConfig())
        fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
        fake_flow = np.zeros((100, 100, 2), dtype=np.float32)

        fake_stream = _FakeFlowStream([
            _FakeFlowFrameResult(
                _FakeFlowTensor(fake_flow),
                _FakeFlowImage(fake_image),
                stream_id=0,
            ),
        ])

        def _factory(**kwargs):
            return fake_stream

        renderer = OverlayRenderer(OverlayConfig())

        records = run_axelera_flow_probe_with_overlay(
            network="edgeflownet-opticalflow",
            source="fakevideo",
            max_frames=1,
            flow_backend=backend,
            stream_factory=_factory,
            overlay_renderer=renderer,
        )

        assert len(records) == 1
        assert records[0].annotated_frame is not None
        assert records[0].annotated_frame.shape == (100, 100, 3)

    def test_flow_probe_overlay_record_to_dict(self):
        from flowsentry.runtime import FlowProbeOverlayRecord

        record = FlowProbeOverlayRecord(
            frame_index=1,
            stream_id=0,
            flow_present=True,
            flow_consistent=True,
            flow_bbox=(10.0, 20.0, 50.0, 60.0),
            flow_region_count=3,
            frame_age_s=0.25,
            annotated_frame=None,
        )

        d = record.to_dict()
        assert d["frame_index"] == 1
        assert d["stream_id"] == 0
        assert d["flow_present"] is True
        assert d["flow_consistent"] is True
        assert d["flow_bbox"] == [10.0, 20.0, 50.0, 60.0]
        assert d["flow_region_count"] == 3
        assert d["frame_age_s"] == 0.25

    def test_run_axelera_flow_probe_with_overlay_drops_stale_frames(self):
        import numpy as np
        from flowsentry.runtime.flow_probe import run_axelera_flow_probe_with_overlay

        backend = AxeleraFlowBackend(config=FlowConfig())
        fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
        fake_flow = np.zeros((100, 100, 2), dtype=np.float32)
        now = time.time()

        fake_stream = _FakeFlowStream([
            _FakeFlowFrameResult(
                _FakeFlowTensor(fake_flow),
                _FakeFlowImage(fake_image),
                stream_id=0,
                src_timestamp=now - 3.0,
            ),
            _FakeFlowFrameResult(
                _FakeFlowTensor(fake_flow),
                _FakeFlowImage(fake_image),
                stream_id=0,
                src_timestamp=now,
            ),
        ])

        def _factory(**kwargs):
            return fake_stream

        records = run_axelera_flow_probe_with_overlay(
            network="edgeflownet-opticalflow",
            source="rtsp://example",
            max_frames=2,
            flow_backend=backend,
            stream_factory=_factory,
            max_frame_age_s=1.0,
        )

        assert fake_stream.stopped is True
        assert len(records) == 1
        assert records[0].frame_index == 2
        assert records[0].frame_age_s is not None
