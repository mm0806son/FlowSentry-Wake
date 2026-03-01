from __future__ import annotations

from dataclasses import dataclass
import json
import sys

from flowsentry.runtime.adapters import AxeleraYoloBackend
from flowsentry.runtime.yolo_probe import (
    YoloProbePreflight,
    YoloProbeRunner,
    YoloProbeRecord,
    list_yolo_network_candidates,
    preflight_axelera_yolo_probe,
    resolve_probe_network_name,
    run_axelera_yolo_probe,
    summarize_probe_records,
    validate_axelera_runtime_env,
    write_probe_jsonl,
    write_probe_summary_json,
)


@dataclass
class _FakeLabel:
    name: str


class _FakeDet:
    def __init__(self, box, score, class_id, label_name=None):
        self.box = box
        self.score = score
        self.class_id = class_id
        if label_name is not None:
            self.label = _FakeLabel(label_name)


class _FakeFrameResult:
    def __init__(self, detections, stream_id=0):
        self.detections = detections
        self.stream_id = stream_id


class _FakeStream:
    def __init__(self, frames):
        self._frames = list(frames)
        self.stopped = False

    def __iter__(self):
        return iter(self._frames)

    def stop(self):
        self.stopped = True


def test_yolo_probe_runner_extracts_primary_person_bbox_and_counts():
    backend = AxeleraYoloBackend(min_confidence=0.5)
    runner = YoloProbeRunner(yolo_backend=backend, max_frames=2)
    stream = _FakeStream(
        [
            _FakeFrameResult([_FakeDet((0, 0, 5, 5), 0.9, 2, "car")], stream_id=1),
            _FakeFrameResult(
                [
                    _FakeDet((10, 10, 20, 20), 0.8, 0, "person"),
                    _FakeDet((5, 5, 30, 35), 0.7, 0, "person"),
                ],
                stream_id=1,
            ),
            _FakeFrameResult([_FakeDet((1, 1, 2, 2), 0.99, 0, "person")], stream_id=1),
        ]
    )

    records = list(runner.run(stream))

    assert len(records) == 2
    assert records[0].person_count == 0
    assert records[0].primary_person_bbox is None
    assert records[1].person_count == 2
    assert records[1].primary_person_bbox == (5.0, 5.0, 30.0, 35.0)
    assert records[1].stream_id == 1


def test_run_axelera_yolo_probe_uses_injected_stream_factory_and_stops_stream():
    backend = AxeleraYoloBackend(min_confidence=0.1)
    fake_stream = _FakeStream([_FakeFrameResult([_FakeDet((1, 2, 3, 4), 0.8, 0, "person")])])
    called = {}

    def _factory(**kwargs):
        called.update(kwargs)
        return fake_stream

    records = run_axelera_yolo_probe(
        network="yolov8s-coco",
        source="fakevideo",
        max_frames=1,
        yolo_backend=backend,
        stream_factory=_factory,
        stream_kwargs={"rtsp_latency": 120},
    )

    assert called["network"] == "yolov8s-coco"
    assert called["sources"] == ["fakevideo"]
    assert called["rtsp_latency"] == 120
    assert records[0].person_bboxes == ((1.0, 2.0, 3.0, 4.0),)
    assert fake_stream.stopped is True


class _FakeInfo:
    def __init__(self):
        self._names = [
            "resnet18-imagenet",
            "yolov8n-output-tensor",
            "yolov5m-v7-coco-tracker",
            "yolov8n-yolov8s",
        ]

    def get_all_yaml_names(self):
        return list(self._names)

    def get_info(self, key):
        if key in self._names:
            class _Resolved:
                yaml_name = key

            return _Resolved()
        raise KeyError(f"Invalid network '{key}'")


def test_list_yolo_network_candidates_filters_yolo_names():
    names = list_yolo_network_candidates(network_yaml_info=_FakeInfo())
    assert names == ["yolov5m-v7-coco-tracker", "yolov8n-output-tensor", "yolov8n-yolov8s"]


def test_resolve_probe_network_name_adds_yolo_suggestions_on_failure():
    try:
        resolve_probe_network_name("yolov8n-coco", network_yaml_info=_FakeInfo())
        assert False, "expected ValueError"
    except ValueError as e:
        msg = str(e)
        assert "yolov8n-coco" in msg
        assert "Available YOLO networks" in msg
        assert "yolov8n-output-tensor" in msg


def test_resolve_probe_network_name_accepts_existing_yaml_path(tmp_path):
    p = tmp_path / "custom-yolo.yaml"
    p.write_text("name: custom\n", encoding="utf-8")
    assert resolve_probe_network_name(str(p), network_yaml_info=_FakeInfo()) == str(p)


def test_validate_axelera_runtime_env_requires_activation():
    try:
        validate_axelera_runtime_env({"LD_LIBRARY_PATH": ""})
        assert False, "expected ValueError"
    except ValueError as e:
        assert "AXELERA_FRAMEWORK" in str(e)

    try:
        validate_axelera_runtime_env({"AXELERA_FRAMEWORK": "/tmp", "LD_LIBRARY_PATH": ""})
        assert False, "expected ValueError"
    except ValueError as e:
        assert "LD_LIBRARY_PATH" in str(e)

    validate_axelera_runtime_env(
        {
            "AXELERA_FRAMEWORK": "/home/orangepi/voyager-sdk",
            "LD_LIBRARY_PATH": "/opt/axelera/runtime-1.5.2-1/lib:/tmp",
        }
    )


def test_preflight_axelera_yolo_probe_validates_env_and_resolves_network():
    preflight = preflight_axelera_yolo_probe(
        network="yolov8n-output-tensor",
        source="fakevideo",
        pipe_type="gst",
        env={
            "AXELERA_FRAMEWORK": "/home/orangepi/voyager-sdk",
            "LD_LIBRARY_PATH": "/opt/axelera/runtime-1.5.2-1/lib:/tmp",
        },
        network_yaml_info=_FakeInfo(),
    )

    assert isinstance(preflight, YoloProbePreflight)
    assert preflight.requested_network == "yolov8n-output-tensor"
    assert preflight.resolved_network == "yolov8n-output-tensor"
    assert preflight.source == "fakevideo"
    assert preflight.pipe_type == "gst"
    assert preflight.to_dict()["resolved_network"] == "yolov8n-output-tensor"


def test_preflight_axelera_yolo_probe_allows_sourceless_check():
    preflight = preflight_axelera_yolo_probe(
        network="yolov8n-output-tensor",
        env={
            "AXELERA_FRAMEWORK": "/home/orangepi/voyager-sdk",
            "LD_LIBRARY_PATH": "/opt/axelera/runtime-1.5.2-1/lib:/tmp",
        },
        network_yaml_info=_FakeInfo(),
    )

    assert preflight.source is None


def test_probe_summary_and_json_outputs(tmp_path):
    records = [
        YoloProbeRecord(1, 0, 0, (), None),
        YoloProbeRecord(2, 0, 2, ((1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0)), (5.0, 6.0, 7.0, 8.0)),
        YoloProbeRecord(3, 1, 1, ((9.0, 10.0, 11.0, 12.0),), (9.0, 10.0, 11.0, 12.0)),
    ]
    summary = summarize_probe_records(records)

    assert summary.frames_processed == 3
    assert summary.frames_with_persons == 2
    assert summary.total_person_detections == 3
    assert summary.max_person_count == 2
    assert summary.first_person_frame_index == 2
    assert summary.stream_ids == (0, 1)

    jsonl_path = write_probe_jsonl(records, tmp_path / "probe" / "events.jsonl")
    summary_path = write_probe_summary_json(summary, tmp_path / "probe" / "summary.json")

    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert first["frame_index"] == 1
    assert first["person_count"] == 0

    summary_obj = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_obj["frames_processed"] == 3
    assert summary_obj["stream_ids"] == [0, 1]


class _FakeImage:
    def __init__(self, data):
        self._data = data

    def asarray(self):
        return self._data


class _FakeFrameResultWithImage:
    def __init__(self, detections, image_data=None, stream_id=0):
        self.detections = detections
        self.image = _FakeImage(image_data) if image_data is not None else None
        self.stream_id = stream_id


class _FakeStreamWithImage:
    def __init__(self, frames):
        self._frames = list(frames)
        self.stopped = False

    def __iter__(self):
        return iter(self._frames)

    def stop(self):
        self.stopped = True


def test_run_axelera_yolo_probe_with_overlay_extracts_records():
    import numpy as np
    from flowsentry.runtime import run_axelera_yolo_probe_with_overlay

    backend = AxeleraYoloBackend(min_confidence=0.1)
    fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
    fake_image[:] = (100, 150, 200)

    fake_stream = _FakeStreamWithImage([
        _FakeFrameResultWithImage(
            [_FakeDet((10, 10, 50, 50), 0.9, 0, "person")],
            image_data=fake_image,
            stream_id=0,
        ),
        _FakeFrameResultWithImage(
            [_FakeDet((20, 20, 60, 60), 0.8, 0, "person")],
            image_data=fake_image,
            stream_id=0,
        ),
    ])

    def _factory(**kwargs):
        return fake_stream

    records = run_axelera_yolo_probe_with_overlay(
        network="yolov8s-coco",
        source="fakevideo",
        max_frames=2,
        yolo_backend=backend,
        stream_factory=_factory,
    )

    assert len(records) == 2
    assert records[0].person_count == 1
    assert records[0].person_bboxes == ((10.0, 10.0, 50.0, 50.0),)
    assert records[0].annotated_frame is None
    assert fake_stream.stopped is True


def test_run_axelera_yolo_probe_with_overlay_renders_frames():
    import numpy as np
    from flowsentry.overlay import OverlayConfig, OverlayRenderer
    from flowsentry.runtime import run_axelera_yolo_probe_with_overlay

    backend = AxeleraYoloBackend(min_confidence=0.1)
    fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
    fake_image[:] = (100, 150, 200)

    fake_stream = _FakeStreamWithImage([
        _FakeFrameResultWithImage(
            [_FakeDet((10, 10, 50, 50), 0.9, 0, "person")],
            image_data=fake_image,
            stream_id=0,
        ),
    ])

    def _factory(**kwargs):
        return fake_stream

    renderer = OverlayRenderer(OverlayConfig())

    records = run_axelera_yolo_probe_with_overlay(
        network="yolov8s-coco",
        source="fakevideo",
        max_frames=1,
        yolo_backend=backend,
        stream_factory=_factory,
        overlay_renderer=renderer,
    )

    assert len(records) == 1
    assert records[0].annotated_frame is not None
    assert records[0].annotated_frame.shape == (100, 100, 3)


def test_run_axelera_yolo_probe_with_overlay_creates_video_writer_with_frame_shape(
    tmp_path, monkeypatch
):
    import numpy as np
    from flowsentry.runtime import run_axelera_yolo_probe_with_overlay

    backend = AxeleraYoloBackend(min_confidence=0.1)
    fake_image = np.zeros((123, 321, 3), dtype=np.uint8)
    fake_stream = _FakeStreamWithImage([
        _FakeFrameResultWithImage(
            [_FakeDet((10, 10, 50, 50), 0.9, 0, "person")],
            image_data=fake_image,
            stream_id=0,
        ),
    ])

    created: dict[str, object] = {}

    class _FakeVideoWriter:
        def __init__(self, path, fourcc, fps, size):
            created["path"] = path
            created["fourcc"] = fourcc
            created["fps"] = fps
            created["size"] = size
            created["writes"] = 0
            created["released"] = False

        def write(self, frame):
            created["writes"] = int(created["writes"]) + 1
            created["last_shape"] = frame.shape

        def release(self):
            created["released"] = True

    class _FakeCv2:
        @staticmethod
        def VideoWriter_fourcc(*args):
            created["fourcc_args"] = args
            return 1234

        @staticmethod
        def VideoWriter(path, fourcc, fps, size):
            return _FakeVideoWriter(path, fourcc, fps, size)

    monkeypatch.setitem(sys.modules, "cv2", _FakeCv2)

    def _factory(**kwargs):
        return fake_stream

    out_path = tmp_path / "out.mp4"
    records = run_axelera_yolo_probe_with_overlay(
        network="yolov8s-coco",
        source="fakevideo",
        max_frames=1,
        yolo_backend=backend,
        stream_factory=_factory,
        video_path=out_path,
    )

    assert len(records) == 1
    assert created["path"] == str(out_path)
    assert created["size"] == (321, 123)
    assert created["fps"] == 30.0
    assert created["fourcc_args"] == ("m", "p", "4", "v")
    assert created["writes"] == 1
    assert created["released"] is True


def test_run_axelera_yolo_probe_with_overlay_display_converts_rgb_to_bgr(monkeypatch):
    import numpy as np
    from flowsentry.runtime import run_axelera_yolo_probe_with_overlay

    backend = AxeleraYoloBackend(min_confidence=0.1)
    rgb_image = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb_image[0, 0] = (255, 0, 0)
    fake_stream = _FakeStreamWithImage([
        _FakeFrameResultWithImage(
            [_FakeDet((1, 1, 2, 2), 0.9, 0, "person")],
            image_data=rgb_image,
            stream_id=0,
        ),
    ])

    captured: dict[str, object] = {}

    class _FakeCv2:
        @staticmethod
        def imshow(_name, frame):
            captured["frame"] = frame.copy()

        @staticmethod
        def waitKey(_delay):
            return -1

    monkeypatch.setitem(sys.modules, "cv2", _FakeCv2)

    def _factory(**_kwargs):
        return fake_stream

    records = run_axelera_yolo_probe_with_overlay(
        network="yolov8s-coco",
        source="fakevideo",
        max_frames=1,
        yolo_backend=backend,
        stream_factory=_factory,
        display=True,
    )

    assert len(records) == 1
    shown = captured["frame"]
    assert shown[0, 0].tolist() == [0, 0, 255]


def test_yolo_probe_overlay_record_to_dict():
    from flowsentry.runtime import YoloProbeOverlayRecord

    record = YoloProbeOverlayRecord(
        frame_index=1,
        stream_id=0,
        person_count=2,
        person_bboxes=((1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0)),
        primary_person_bbox=(1.0, 2.0, 3.0, 4.0),
        annotated_frame=None,
    )

    d = record.to_dict()
    assert d["frame_index"] == 1
    assert d["stream_id"] == 0
    assert d["person_count"] == 2
    assert d["person_bboxes"] == [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    assert d["primary_person_bbox"] == [1.0, 2.0, 3.0, 4.0]
