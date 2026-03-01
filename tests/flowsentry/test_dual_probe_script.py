from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import run_flowsentry_dual_probe as dual_probe_script


class TestDualProbeScriptMain:
    def test_main_defaults_to_rtsp100_and_fps7(self, monkeypatch, capsys):
        import flowsentry.runtime as runtime_module

        called: dict[str, object] = {}

        def _fake_preflight(**kwargs):
            return SimpleNamespace(
                requested_network=kwargs["network"],
                resolved_network=f"resolved::{kwargs['network']}",
                source=kwargs.get("source"),
                pipe_type=kwargs.get("pipe_type"),
            )

        def _fake_run_dual_probe(**kwargs):
            called.update(kwargs)
            return []

        monkeypatch.setattr(runtime_module, "preflight_axelera_flow_probe", _fake_preflight)
        monkeypatch.setattr(runtime_module, "preflight_axelera_yolo_probe", _fake_preflight)
        monkeypatch.setattr(runtime_module, "list_flow_network_candidates", lambda: ("edgeflownet-opticalflow-raw",))
        monkeypatch.setattr(dual_probe_script, "run_dual_probe", _fake_run_dual_probe)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_flowsentry_dual_probe.py",
                "edgeflownet-opticalflow",
                "yolov8s-coco",
                "fakevideo",
                "--summary-only",
                "--frames",
                "1",
            ],
        )

        dual_probe_script.main()

        out = capsys.readouterr().out
        assert "Summary: frames=0" in out
        assert called["flow_network"] == "resolved::edgeflownet-opticalflow-raw"
        assert called["yolo_network"] == "resolved::yolov8s-coco"
        assert called["stream_kwargs"] == {"rtsp_latency": 100, "specified_frame_rate": 7}

    def test_main_can_disable_dmabuf(self, monkeypatch):
        import flowsentry.runtime as runtime_module

        def _fake_preflight(**kwargs):
            return SimpleNamespace(
                requested_network=kwargs["network"],
                resolved_network=kwargs["network"],
                source=kwargs.get("source"),
                pipe_type=kwargs.get("pipe_type"),
            )

        def _fake_run_dual_probe(**_kwargs):
            return []

        monkeypatch.setattr(runtime_module, "preflight_axelera_flow_probe", _fake_preflight)
        monkeypatch.setattr(runtime_module, "preflight_axelera_yolo_probe", _fake_preflight)
        monkeypatch.setattr(runtime_module, "list_flow_network_candidates", lambda: ("edgeflownet-opticalflow-raw",))
        monkeypatch.setattr(dual_probe_script, "run_dual_probe", _fake_run_dual_probe)
        monkeypatch.delenv("AXELERA_USE_DMABUF", raising=False)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_flowsentry_dual_probe.py",
                "edgeflownet-opticalflow",
                "yolov8s-coco",
                "fakevideo",
                "--summary-only",
                "--frames",
                "1",
                "--no-use-dmabuf",
            ],
        )

        dual_probe_script.main()
        assert os.environ["AXELERA_USE_DMABUF"] == "0"

    def test_main_display_flag_is_forwarded(self, monkeypatch):
        import flowsentry.runtime as runtime_module

        called: dict[str, object] = {}

        def _fake_preflight(**kwargs):
            return SimpleNamespace(
                requested_network=kwargs["network"],
                resolved_network=kwargs["network"],
                source=kwargs.get("source"),
                pipe_type=kwargs.get("pipe_type"),
            )

        def _fake_run_dual_probe(**kwargs):
            called.update(kwargs)
            return []

        monkeypatch.setattr(runtime_module, "preflight_axelera_flow_probe", _fake_preflight)
        monkeypatch.setattr(runtime_module, "preflight_axelera_yolo_probe", _fake_preflight)
        monkeypatch.setattr(runtime_module, "list_flow_network_candidates", lambda: ("edgeflownet-opticalflow-raw",))
        monkeypatch.setattr(dual_probe_script, "run_dual_probe", _fake_run_dual_probe)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_flowsentry_dual_probe.py",
                "edgeflownet-opticalflow",
                "yolov8s-coco",
                "fakevideo",
                "--summary-only",
                "--frames",
                "1",
                "--display",
            ],
        )

        dual_probe_script.main()
        assert called["display"] is True
