#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SCENARIO_TYPES = ("normal", "false_positive", "adversarial")


@dataclass(frozen=True)
class ScenarioManifest:
    scenario_type: str
    expected_result: str
    actual_result: str
    video_path: str | None
    summary: dict[str, Any]
    timestamp: str
    duration_seconds: float
    frames_processed: int
    alarm_triggered: bool
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_type": self.scenario_type,
            "expected_result": self.expected_result,
            "actual_result": self.actual_result,
            "video_path": self.video_path,
            "summary": self.summary,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "frames_processed": self.frames_processed,
            "alarm_triggered": self.alarm_triggered,
            "notes": self.notes,
        }


def get_expected_result(scenario_type: str) -> str:
    expected = {
        "normal": "alarm_triggered",
        "false_positive": "no_alarm",
        "adversarial": "alarm_triggered",
    }
    return expected.get(scenario_type, "unknown")


def determine_actual_result(alarm_triggered: bool) -> str:
    return "alarm_triggered" if alarm_triggered else "no_alarm"


def run_scenario(
    *,
    flow_network: str,
    yolo_network: str,
    source: str,
    scenario_type: str,
    duration_seconds: float,
    output_dir: Path,
    pipe_type: str = "gst",
    stream_kwargs: dict[str, Any] | None = None,
    save_video: bool = True,
    notes: str | None = None,
    stream_factory: Any | None = None,
    flow_magnitude_threshold: float | None = None,
    flow_min_region_area: int | None = None,
) -> ScenarioManifest:
    from scripts.run_flowsentry_dual_probe import (
        run_dual_probe,
        summarize_dual_probe_records,
    )
    from flowsentry.overlay import OverlayConfig, OverlayRenderer

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario_dir = output_dir / f"{timestamp}_{scenario_type}"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    video_path = None
    overlay_renderer = OverlayRenderer(OverlayConfig())

    if save_video:
        video_path = scenario_dir / "video.mp4"

    start_time = time.time()
    max_frames = int(duration_seconds * 30)

    records = run_dual_probe(
        flow_network=flow_network,
        yolo_network=yolo_network,
        source=source,
        pipe_type=pipe_type,
        max_frames=max_frames,
        stream_kwargs=stream_kwargs,
        overlay_renderer=overlay_renderer,
        video_path=video_path,
        display=False,
        stream_factory=stream_factory,
        flow_magnitude_threshold=flow_magnitude_threshold,
        flow_min_region_area=flow_min_region_area,
    )

    elapsed = time.time() - start_time
    summary = summarize_dual_probe_records(records)

    actual_result = determine_actual_result(summary.alarm_triggered)
    expected_result = get_expected_result(scenario_type)

    manifest = ScenarioManifest(
        scenario_type=scenario_type,
        expected_result=expected_result,
        actual_result=actual_result,
        video_path=str(video_path) if video_path and video_path.exists() else None,
        summary=summary.to_dict(),
        timestamp=timestamp,
        duration_seconds=round(elapsed, 2),
        frames_processed=summary.frames_processed,
        alarm_triggered=summary.alarm_triggered,
        notes=notes,
    )

    manifest_path = scenario_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return manifest


def main() -> None:
    from flowsentry.config import FlowConfig

    default_flow_config = FlowConfig()
    parser = argparse.ArgumentParser(description="FlowSentry scenario validation runner")
    parser.add_argument("flow_network", help="Flow network name")
    parser.add_argument("yolo_network", help="YOLO network name")
    parser.add_argument("source", help="Input source (camera or RTSP URL)")
    parser.add_argument(
        "--scenario-type",
        choices=SCENARIO_TYPES,
        required=True,
        help="Scenario type to validate",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Recording duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/flowsentry/scenarios",
        help="Output directory for scenario artifacts",
    )
    parser.add_argument(
        "--pipe",
        dest="pipe_type",
        default="gst",
        help="Pipeline type (default: gst)",
    )
    parser.add_argument(
        "--rtsp-latency",
        type=int,
        help="Optional RTSP latency (ms)",
    )
    parser.add_argument(
        "--magnitude-threshold",
        type=float,
        default=default_flow_config.mask_magnitude_threshold,
        help="Flow magnitude threshold for region detection",
    )
    parser.add_argument(
        "--min-region-area",
        type=int,
        default=default_flow_config.mask_min_region_area,
        help="Minimum region area in pixels",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Do not save video",
    )
    parser.add_argument(
        "--notes",
        type=str,
        help="Additional notes for the manifest",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stream_kwargs = {}
    if args.rtsp_latency is not None:
        stream_kwargs["rtsp_latency"] = args.rtsp_latency

    print(f"Starting scenario: {args.scenario_type}")
    print(f"Duration: {args.duration}s")
    print(f"Source: {args.source}")

    manifest = run_scenario(
        flow_network=args.flow_network,
        yolo_network=args.yolo_network,
        source=args.source,
        scenario_type=args.scenario_type,
        duration_seconds=args.duration,
        output_dir=output_dir,
        pipe_type=args.pipe_type,
        stream_kwargs=stream_kwargs,
        save_video=not args.no_video,
        notes=args.notes,
        flow_magnitude_threshold=args.magnitude_threshold,
        flow_min_region_area=args.min_region_area,
    )

    print("\n=== Scenario Complete ===")
    print(f"Type: {manifest.scenario_type}")
    print(f"Expected: {manifest.expected_result}")
    print(f"Actual: {manifest.actual_result}")
    print(f"Frames: {manifest.frames_processed}")
    print(f"Duration: {manifest.duration_seconds}s")
    print(f"Alarm triggered: {manifest.alarm_triggered}")
    if manifest.video_path:
        print(f"Video: {manifest.video_path}")

    if manifest.expected_result == manifest.actual_result:
        print("\n[PASS] Scenario validation passed!")
        return 0
    else:
        print(f"\n[FAIL] Scenario validation failed! Expected {manifest.expected_result}, got {manifest.actual_result}")
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
