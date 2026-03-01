from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import run_flowsentry_flow_probe as flow_probe_script


class _FakeBackend:
    def __init__(self, config, **kwargs):
        self.config = config
        self.kwargs = kwargs


class TestFlowProbeScriptMain:
    def test_main_non_overlay_uses_inference_core_and_prints_latency(self, monkeypatch, capsys):
        from flowsentry.runtime import FlowProbeLatencySummary, FlowProbeRecord
        import flowsentry.runtime as runtime_module
        import flowsentry.runtime.adapters as adapters_module

        called: dict[str, object] = {}

        def _fake_run_inference_core(**kwargs):
            called.update(kwargs)
            records = [
                FlowProbeRecord(
                    frame_index=1,
                    stream_id=0,
                    flow_present=False,
                    flow_consistent=False,
                    flow_bbox=None,
                    flow_region_count=0,
                )
            ]
            latency = FlowProbeLatencySummary(
                sampled_frames=1,
                p50_ms=12.34,
                p95_ms=23.45,
                max_ms=34.56,
            )
            return records, latency

        monkeypatch.setattr(runtime_module, "run_axelera_flow_probe_inference_core", _fake_run_inference_core)
        monkeypatch.setattr(adapters_module, "AxeleraFlowBackend", _FakeBackend)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_flowsentry_flow_probe.py",
                "edgeflownet-opticalflow",
                "fakevideo",
                "--summary-only",
                "--frames",
                "1",
                "--rtsp-latency",
                "100",
            ],
        )

        flow_probe_script.main()

        out = capsys.readouterr().out
        assert "LatencySummary: sampled_frames=1, p50_ms=12.34, p95_ms=23.45, max_ms=34.56" in out
        assert called["network"] == "edgeflownet-opticalflow"
        assert called["source"] == "fakevideo"
        assert called["max_frames"] == 1
        assert called["display_enabled"] is False
        assert called["enable_flow_extract"] is True
        assert called["stream_kwargs"] == {
            "rtsp_latency": 100,
            "specified_frame_rate": 7,
        }

    def test_main_defaults_to_dmabuf_enabled(self, monkeypatch):
        from flowsentry.runtime import FlowProbeLatencySummary, FlowProbeRecord
        import flowsentry.runtime as runtime_module
        import flowsentry.runtime.adapters as adapters_module

        def _fake_run_inference_core(**_kwargs):
            records = [
                FlowProbeRecord(
                    frame_index=1,
                    stream_id=0,
                    flow_present=False,
                    flow_consistent=False,
                    flow_bbox=None,
                    flow_region_count=0,
                )
            ]
            return records, FlowProbeLatencySummary(
                sampled_frames=1,
                p50_ms=1.0,
                p95_ms=1.0,
                max_ms=1.0,
            )

        monkeypatch.setattr(runtime_module, "run_axelera_flow_probe_inference_core", _fake_run_inference_core)
        monkeypatch.setattr(adapters_module, "AxeleraFlowBackend", _FakeBackend)
        monkeypatch.setenv("AXELERA_USE_DMABUF", "0")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_flowsentry_flow_probe.py",
                "edgeflownet-opticalflow",
                "fakevideo",
                "--summary-only",
                "--frames",
                "1",
            ],
        )

        flow_probe_script.main()

        assert "AXELERA_USE_DMABUF" not in os.environ

    def test_main_can_disable_dmabuf(self, monkeypatch):
        from flowsentry.runtime import FlowProbeLatencySummary, FlowProbeRecord
        import flowsentry.runtime as runtime_module
        import flowsentry.runtime.adapters as adapters_module

        def _fake_run_inference_core(**_kwargs):
            records = [
                FlowProbeRecord(
                    frame_index=1,
                    stream_id=0,
                    flow_present=False,
                    flow_consistent=False,
                    flow_bbox=None,
                    flow_region_count=0,
                )
            ]
            return records, FlowProbeLatencySummary(
                sampled_frames=1,
                p50_ms=1.0,
                p95_ms=1.0,
                max_ms=1.0,
            )

        monkeypatch.setattr(runtime_module, "run_axelera_flow_probe_inference_core", _fake_run_inference_core)
        monkeypatch.setattr(adapters_module, "AxeleraFlowBackend", _FakeBackend)
        monkeypatch.delenv("AXELERA_USE_DMABUF", raising=False)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_flowsentry_flow_probe.py",
                "edgeflownet-opticalflow",
                "fakevideo",
                "--summary-only",
                "--frames",
                "1",
                "--no-use-dmabuf",
            ],
        )

        flow_probe_script.main()

        assert os.environ["AXELERA_USE_DMABUF"] == "0"

    def test_main_overlay_uses_overlay_runner_and_emits_empty_latency(self, monkeypatch, capsys):
        from flowsentry.runtime import FlowProbeOverlayRecord
        import flowsentry.runtime as runtime_module
        import flowsentry.runtime.adapters as adapters_module

        called: dict[str, object] = {}

        def _fail_inference_core(**_kwargs):
            raise AssertionError("inference core path should not run in overlay mode")

        def _fake_run_overlay(**kwargs):
            called.update(kwargs)
            return [
                FlowProbeOverlayRecord(
                    frame_index=1,
                    stream_id=0,
                    flow_present=True,
                    flow_consistent=False,
                    flow_bbox=(1.0, 2.0, 3.0, 4.0),
                    flow_region_count=1,
                    frame_age_s=0.12,
                    annotated_frame=None,
                )
            ]

        monkeypatch.setattr(runtime_module, "run_axelera_flow_probe_inference_core", _fail_inference_core)
        monkeypatch.setattr(runtime_module, "run_axelera_flow_probe_with_overlay", _fake_run_overlay)
        monkeypatch.setattr(adapters_module, "AxeleraFlowBackend", _FakeBackend)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_flowsentry_flow_probe.py",
                "edgeflownet-opticalflow",
                "fakevideo",
                "--summary-only",
                "--overlay",
                "--frames",
                "1",
            ],
        )

        flow_probe_script.main()

        out = capsys.readouterr().out
        assert "LatencySummary: sampled_frames=1, p50_ms=120.0, p95_ms=120.0, max_ms=120.0" in out
        assert called["network"] == "edgeflownet-opticalflow"
        assert called["source"] == "fakevideo"
        assert called["max_frames"] == 1
        assert called["display"] is True
        assert called["use_native_display"] is True

    def test_main_overlay_can_disable_native_display(self, monkeypatch):
        from flowsentry.runtime import FlowProbeOverlayRecord
        import flowsentry.runtime as runtime_module
        import flowsentry.runtime.adapters as adapters_module

        called: dict[str, object] = {}

        def _fail_inference_core(**_kwargs):
            raise AssertionError("inference core path should not run in overlay mode")

        def _fake_run_overlay(**kwargs):
            called.update(kwargs)
            return [
                FlowProbeOverlayRecord(
                    frame_index=1,
                    stream_id=0,
                    flow_present=False,
                    flow_consistent=False,
                    flow_bbox=None,
                    flow_region_count=0,
                    frame_age_s=0.12,
                    annotated_frame=None,
                )
            ]

        monkeypatch.setattr(runtime_module, "run_axelera_flow_probe_inference_core", _fail_inference_core)
        monkeypatch.setattr(runtime_module, "run_axelera_flow_probe_with_overlay", _fake_run_overlay)
        monkeypatch.setattr(adapters_module, "AxeleraFlowBackend", _FakeBackend)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_flowsentry_flow_probe.py",
                "edgeflownet-opticalflow",
                "fakevideo",
                "--summary-only",
                "--overlay",
                "--no-native-display",
                "--frames",
                "1",
            ],
        )

        flow_probe_script.main()

        assert called["use_native_display"] is False

    def test_main_debug_flow_tensor_passes_debug_callback(self, monkeypatch):
        from flowsentry.runtime import FlowProbeLatencySummary, FlowProbeRecord
        import flowsentry.runtime as runtime_module
        import flowsentry.runtime.adapters as adapters_module

        created: dict[str, object] = {}

        class _CaptureBackend:
            def __init__(self, config, **kwargs):
                created["config"] = config
                created["kwargs"] = kwargs

        def _fake_run_inference_core(**_kwargs):
            records = [
                FlowProbeRecord(
                    frame_index=1,
                    stream_id=0,
                    flow_present=False,
                    flow_consistent=False,
                    flow_bbox=None,
                    flow_region_count=0,
                )
            ]
            return records, FlowProbeLatencySummary(
                sampled_frames=1,
                p50_ms=1.0,
                p95_ms=1.0,
                max_ms=1.0,
            )

        monkeypatch.setattr(runtime_module, "run_axelera_flow_probe_inference_core", _fake_run_inference_core)
        monkeypatch.setattr(adapters_module, "AxeleraFlowBackend", _CaptureBackend)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_flowsentry_flow_probe.py",
                "edgeflownet-opticalflow",
                "fakevideo",
                "--summary-only",
                "--frames",
                "1",
                "--debug-flow-tensor",
            ],
        )

        flow_probe_script.main()

        assert "kwargs" in created
        cb = created["kwargs"]["debug_callback"]
        assert callable(cb)
