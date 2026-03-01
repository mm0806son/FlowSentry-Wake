from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest


class TestScenarioManifest:
    def test_scenario_manifest_to_dict(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        
        from flowsentry_scenario_runner import ScenarioManifest

        manifest = ScenarioManifest(
            scenario_type="normal",
            expected_result="alarm_triggered",
            actual_result="alarm_triggered",
            video_path="/path/to/video.mp4",
            summary={"frames_processed": 100},
            timestamp="20250224_120000",
            duration_seconds=10.5,
            frames_processed=315,
            alarm_triggered=True,
            notes="Test scenario",
        )

        d = manifest.to_dict()
        assert d["scenario_type"] == "normal"
        assert d["expected_result"] == "alarm_triggered"
        assert d["actual_result"] == "alarm_triggered"
        assert d["video_path"] == "/path/to/video.mp4"
        assert d["alarm_triggered"] is True
        assert d["notes"] == "Test scenario"

    def test_get_expected_result(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        
        from flowsentry_scenario_runner import get_expected_result

        assert get_expected_result("normal") == "alarm_triggered"
        assert get_expected_result("false_positive") == "no_alarm"
        assert get_expected_result("adversarial") == "alarm_triggered"

    def test_determine_actual_result(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        
        from flowsentry_scenario_runner import determine_actual_result

        assert determine_actual_result(True) == "alarm_triggered"
        assert determine_actual_result(False) == "no_alarm"


class _FakeImage:
    def __init__(self, data):
        self._data = data

    def asarray(self):
        return self._data


@dataclass
class _FakeFlowTensor:
    data: np.ndarray

    def numpy(self):
        return self.data


@dataclass
class _FakeDet:
    box: tuple[float, float, float, float]
    score: float
    class_id: int

    @property
    def label(self):
        class _Label:
            name = "person"
        return _Label()


@dataclass
class _FakeFlowFrameResult:
    tensor: _FakeFlowTensor
    image: _FakeImage
    stream_id: int = 0


@dataclass
class _FakeYoloFrameResult:
    detections: list[_FakeDet]
    image: _FakeImage
    stream_id: int = 0


class _FakeDualStream:
    def __init__(self, frames):
        self._frames = list(frames)
        self.stopped = False

    def __iter__(self):
        return iter(self._frames)

    def stop(self):
        self.stopped = True


class TestRunScenario:
    def test_run_scenario_creates_manifest(self, tmp_path: Path):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        
        from flowsentry_scenario_runner import run_scenario

        fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
        fake_flow = np.zeros((100, 100, 2), dtype=np.float32)

        flow_frames = [
            _FakeFlowFrameResult(
                _FakeFlowTensor(fake_flow),
                _FakeImage(fake_image),
                stream_id=0,
            )
            for _ in range(5)
        ]

        yolo_frames = [
            _FakeYoloFrameResult(
                [_FakeDet((10, 10, 50, 50), 0.9, 0)],
                _FakeImage(fake_image),
                stream_id=0,
            )
            for _ in range(5)
        ]

        call_count = [0]

        def _factory(**kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                return _FakeDualStream(flow_frames)
            else:
                return _FakeDualStream(yolo_frames)

        manifest = run_scenario(
            flow_network="edgeflownet-opticalflow",
            yolo_network="yolov8s-coco",
            source="fakevideo",
            scenario_type="normal",
            duration_seconds=0.5,
            output_dir=tmp_path,
            save_video=False,
            stream_factory=_factory,
        )

        assert manifest.scenario_type == "normal"
        assert manifest.expected_result == "alarm_triggered"
        assert manifest.actual_result in ("alarm_triggered", "no_alarm")
        assert manifest.frames_processed == 5

        scenario_dirs = list(tmp_path.iterdir())
        assert len(scenario_dirs) == 1
        assert scenario_dirs[0].name.endswith("_normal")

        manifest_path = scenario_dirs[0] / "manifest.json"
        assert manifest_path.exists()

        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["scenario_type"] == "normal"

    def test_run_scenario_with_alarm(self, tmp_path: Path):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        
        from flowsentry_scenario_runner import run_scenario

        fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
        fake_flow_with_motion = np.ones((100, 100, 2), dtype=np.float32) * 10

        flow_frames = [
            _FakeFlowFrameResult(
                _FakeFlowTensor(fake_flow_with_motion),
                _FakeImage(fake_image),
                stream_id=0,
            )
            for _ in range(10)
        ]

        yolo_frames = [
            _FakeYoloFrameResult(
                [_FakeDet((10, 10, 50, 50), 0.9, 0)],
                _FakeImage(fake_image),
                stream_id=0,
            )
            for _ in range(10)
        ]

        call_count = [0]

        def _factory(**kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                return _FakeDualStream(flow_frames)
            else:
                return _FakeDualStream(yolo_frames)

        manifest = run_scenario(
            flow_network="edgeflownet-opticalflow",
            yolo_network="yolov8s-coco",
            source="fakevideo",
            scenario_type="normal",
            duration_seconds=0.5,
            output_dir=tmp_path,
            save_video=False,
            stream_factory=_factory,
        )

        assert manifest.scenario_type == "normal"
        assert manifest.expected_result == "alarm_triggered"
