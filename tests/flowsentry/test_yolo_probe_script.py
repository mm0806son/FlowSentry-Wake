from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import run_flowsentry_yolo_probe as yolo_probe_script


class _FakeBackend:
    def __init__(self, min_confidence: float):
        self.min_confidence = min_confidence


class _FakeOverlayConfig:
    pass


class _FakeOverlayRenderer:
    def __init__(self, _config):
        self.config = _config


class TestYoloProbeScriptMain:
    def test_main_overlay_save_video_defaults_display_off(self, monkeypatch, tmp_path):
        import cv2
        import flowsentry.overlay as overlay_module
        import flowsentry.runtime as runtime_module
        import flowsentry.runtime.adapters as adapters_module

        called: dict[str, object] = {}

        def _fail_plain_path(**_kwargs):
            raise AssertionError("plain probe path should not run in overlay mode")

        def _fake_run_overlay(**kwargs):
            called.update(kwargs)
            return [
                SimpleNamespace(
                    frame_index=1,
                    stream_id=0,
                    person_count=1,
                    person_bboxes=((1.0, 2.0, 3.0, 4.0),),
                    primary_person_bbox=(1.0, 2.0, 3.0, 4.0),
                )
            ]

        monkeypatch.setattr(runtime_module, "run_axelera_yolo_probe", _fail_plain_path)
        monkeypatch.setattr(runtime_module, "run_axelera_yolo_probe_with_overlay", _fake_run_overlay)
        monkeypatch.setattr(adapters_module, "AxeleraYoloBackend", _FakeBackend)
        monkeypatch.setattr(overlay_module, "OverlayConfig", _FakeOverlayConfig)
        monkeypatch.setattr(overlay_module, "OverlayRenderer", _FakeOverlayRenderer)
        monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

        out_video = tmp_path / "yolo_overlay.mp4"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_flowsentry_yolo_probe.py",
                "yolov8s-coco",
                "fakevideo",
                "--summary-only",
                "--frames",
                "1",
                "--overlay",
                "--save-video",
                str(out_video),
            ],
        )

        yolo_probe_script.main()

        assert called["video_path"] == str(out_video)
        assert called["display"] is False

    def test_main_overlay_save_video_can_enable_display(self, monkeypatch, tmp_path):
        import cv2
        import flowsentry.overlay as overlay_module
        import flowsentry.runtime as runtime_module
        import flowsentry.runtime.adapters as adapters_module

        called: dict[str, object] = {}

        def _fake_run_overlay(**kwargs):
            called.update(kwargs)
            return [
                SimpleNamespace(
                    frame_index=1,
                    stream_id=0,
                    person_count=0,
                    person_bboxes=(),
                    primary_person_bbox=None,
                )
            ]

        monkeypatch.setattr(runtime_module, "run_axelera_yolo_probe_with_overlay", _fake_run_overlay)
        monkeypatch.setattr(adapters_module, "AxeleraYoloBackend", _FakeBackend)
        monkeypatch.setattr(overlay_module, "OverlayConfig", _FakeOverlayConfig)
        monkeypatch.setattr(overlay_module, "OverlayRenderer", _FakeOverlayRenderer)
        monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

        out_video = tmp_path / "yolo_overlay_with_display.mp4"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_flowsentry_yolo_probe.py",
                "yolov8s-coco",
                "fakevideo",
                "--summary-only",
                "--frames",
                "1",
                "--save-video",
                str(out_video),
                "--display",
            ],
        )

        yolo_probe_script.main()

        assert called["video_path"] == str(out_video)
        assert called["display"] is True
