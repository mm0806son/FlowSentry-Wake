from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

from flowsentry.config import TriageConfig
from flowsentry.runtime.adapters import AxeleraYoloBackend
from flowsentry.runtime.triage_replay import (
    TriageReplayRecord,
    TriageReplaySummary,
    replay_yolo_probe_to_triage,
    summarize_triage_replay,
    write_triage_replay_jsonl,
    write_triage_replay_summary_json,
)
from flowsentry.runtime.yolo_probe import (
    YoloProbePreflight,
    YoloProbeRecord,
    YoloProbeSummary,
    preflight_axelera_yolo_probe,
    run_axelera_yolo_probe,
    summarize_probe_records,
    write_probe_jsonl,
    write_probe_summary_json,
)


@dataclass(frozen=True)
class Stage2AcceptancePaths:
    root_dir: Path
    probe_jsonl: Path
    probe_summary_json: Path
    triage_jsonl: Path
    triage_summary_json: Path
    manifest_json: Path


@dataclass(frozen=True)
class Stage2AcceptanceResult:
    preflight: YoloProbePreflight
    probe_records: list[YoloProbeRecord]
    probe_summary: YoloProbeSummary
    triage_records: list[TriageReplayRecord]
    triage_summary: TriageReplaySummary
    paths: Stage2AcceptancePaths


def _safe_tag(tag: str) -> str:
    safe = "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in tag.strip())
    return safe or "run"


def default_stage2_acceptance_dir(
    *,
    output_root: str | Path = "artifacts/flowsentry/stage2_acceptance",
    tag: str | None = None,
) -> Path:
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{_safe_tag(tag)}" if tag else ""
    return Path(output_root) / f"{ts}{suffix}"


def build_stage2_acceptance_paths(root_dir: str | Path) -> Stage2AcceptancePaths:
    root = Path(root_dir)
    return Stage2AcceptancePaths(
        root_dir=root,
        probe_jsonl=root / "probe_events.jsonl",
        probe_summary_json=root / "probe_summary.json",
        triage_jsonl=root / "triage_events.jsonl",
        triage_summary_json=root / "triage_summary.json",
        manifest_json=root / "manifest.json",
    )


def write_stage2_acceptance_manifest(
    result: Stage2AcceptanceResult,
    *,
    network: str,
    source: str,
    pipe_type: str,
    max_frames: int | None,
    person_conf: float,
    triage_config: TriageConfig,
    stream_kwargs: dict[str, Any] | None = None,
) -> Path:
    manifest = {
        "stage": "stage2_real_yolo_integration_acceptance",
        "inputs": {
            "network": network,
            "source": source,
            "pipe_type": pipe_type,
            "max_frames": max_frames,
            "person_conf": person_conf,
            "stream_kwargs": dict(stream_kwargs or {}),
            "triage": {
                "consistency_frames_threshold": triage_config.flow.consistency_frames_threshold,
                "iou_threshold": triage_config.fusion.iou_threshold,
                "no_motion_reset_frames": triage_config.runtime.no_motion_reset_frames,
            },
        },
        "preflight": result.preflight.to_dict(),
        "probe_summary": result.probe_summary.to_dict(),
        "triage_summary": result.triage_summary.to_dict(),
        "artifacts": {
            "probe_jsonl": str(result.paths.probe_jsonl),
            "probe_summary_json": str(result.paths.probe_summary_json),
            "triage_jsonl": str(result.paths.triage_jsonl),
            "triage_summary_json": str(result.paths.triage_summary_json),
        },
    }
    result.paths.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    result.paths.manifest_json.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result.paths.manifest_json


def run_stage2_acceptance(
    *,
    network: str,
    source: str,
    pipe_type: str = "gst",
    max_frames: int | None = 60,
    person_conf: float = 0.25,
    output_dir: str | Path,
    triage_config: TriageConfig | None = None,
    stream_kwargs: dict[str, Any] | None = None,
    preflight_fn: Callable[..., YoloProbePreflight] = preflight_axelera_yolo_probe,
    probe_runner: Callable[..., list[YoloProbeRecord]] = run_axelera_yolo_probe,
) -> Stage2AcceptanceResult:
    cfg = triage_config or TriageConfig()
    paths = build_stage2_acceptance_paths(output_dir)
    paths.root_dir.mkdir(parents=True, exist_ok=True)

    preflight = preflight_fn(network=network, source=source, pipe_type=pipe_type)

    probe_records = probe_runner(
        network=network,
        source=source,
        pipe_type=pipe_type,
        max_frames=max_frames,
        yolo_backend=AxeleraYoloBackend(min_confidence=person_conf),
        stream_kwargs=stream_kwargs,
    )
    probe_summary = summarize_probe_records(probe_records)
    write_probe_jsonl(probe_records, paths.probe_jsonl)
    write_probe_summary_json(probe_summary, paths.probe_summary_json)

    triage_records = replay_yolo_probe_to_triage(probe_records, triage_config=cfg)
    triage_summary = summarize_triage_replay(triage_records)
    write_triage_replay_jsonl(triage_records, paths.triage_jsonl)
    write_triage_replay_summary_json(triage_summary, paths.triage_summary_json)

    result = Stage2AcceptanceResult(
        preflight=preflight,
        probe_records=probe_records,
        probe_summary=probe_summary,
        triage_records=triage_records,
        triage_summary=triage_summary,
        paths=paths,
    )
    write_stage2_acceptance_manifest(
        result,
        network=network,
        source=source,
        pipe_type=pipe_type,
        max_frames=max_frames,
        person_conf=person_conf,
        triage_config=cfg,
        stream_kwargs=stream_kwargs,
    )
    return result
