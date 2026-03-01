from __future__ import annotations

import json

from flowsentry.config import TriageConfig
from flowsentry.runtime.stage2_acceptance import (
    build_stage2_acceptance_paths,
    default_stage2_acceptance_dir,
    run_stage2_acceptance,
)
from flowsentry.runtime.yolo_probe import YoloProbePreflight, YoloProbeRecord


def test_default_stage2_acceptance_dir_appends_sanitized_tag():
    p = default_stage2_acceptance_dir(output_root="/tmp/x", tag="rtsp lobby/cam#1")
    assert str(p).startswith("/tmp/x/")
    assert p.name.endswith("_rtsp_lobby_cam_1")


def test_run_stage2_acceptance_writes_all_artifacts_and_manifest(tmp_path):
    called = {}

    def _preflight_fn(**kwargs):
        called["preflight"] = kwargs
        return YoloProbePreflight(
            requested_network=kwargs["network"],
            resolved_network="resolved-yolo",
            source=kwargs["source"],
            pipe_type=kwargs["pipe_type"],
        )

    def _probe_runner(**kwargs):
        called["probe"] = kwargs
        return [
            YoloProbeRecord(1, 0, 0, (), None),
            YoloProbeRecord(2, 0, 1, ((10.0, 10.0, 50.0, 50.0),), (10.0, 10.0, 50.0, 50.0)),
            YoloProbeRecord(3, 0, 1, ((10.0, 10.0, 50.0, 50.0),), (10.0, 10.0, 50.0, 50.0)),
        ]

    cfg = TriageConfig()
    cfg.flow.consistency_frames_threshold = 2

    out_dir = tmp_path / "acceptance"
    result = run_stage2_acceptance(
        network="yolo-test",
        source="fakevideo",
        max_frames=3,
        person_conf=0.42,
        output_dir=out_dir,
        triage_config=cfg,
        stream_kwargs={"rtsp_latency": 200},
        preflight_fn=_preflight_fn,
        probe_runner=_probe_runner,
    )

    assert called["preflight"]["network"] == "yolo-test"
    assert called["probe"]["network"] == "yolo-test"
    assert called["probe"]["source"] == "fakevideo"
    assert called["probe"]["max_frames"] == 3
    assert called["probe"]["stream_kwargs"] == {"rtsp_latency": 200}
    assert "yolo_backend" in called["probe"]
    assert result.probe_summary.frames_processed == 3
    assert result.probe_summary.frames_with_persons == 2
    assert result.triage_summary.frames_processed == 3
    assert result.triage_summary.alarms_triggered == 1

    paths = build_stage2_acceptance_paths(out_dir)
    for p in (
        paths.probe_jsonl,
        paths.probe_summary_json,
        paths.triage_jsonl,
        paths.triage_summary_json,
        paths.manifest_json,
    ):
        assert p.exists(), f"missing artifact: {p}"

    manifest = json.loads(paths.manifest_json.read_text(encoding="utf-8"))
    assert manifest["inputs"]["network"] == "yolo-test"
    assert manifest["inputs"]["source"] == "fakevideo"
    assert manifest["inputs"]["person_conf"] == 0.42
    assert manifest["inputs"]["stream_kwargs"] == {"rtsp_latency": 200}
    assert manifest["preflight"]["resolved_network"] == "resolved-yolo"
    assert manifest["probe_summary"]["frames_processed"] == 3
    assert manifest["triage_summary"]["alarms_triggered"] == 1
