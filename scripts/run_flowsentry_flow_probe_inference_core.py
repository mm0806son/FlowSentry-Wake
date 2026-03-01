#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    from flowsentry.config import FlowConfig
    from flowsentry.runtime import (
        list_flow_network_candidates,
        preflight_axelera_flow_probe,
        run_axelera_flow_probe_inference_core,
        summarize_flow_probe_records,
        write_flow_probe_jsonl,
        write_flow_probe_summary_json,
    )

    default_flow_config = FlowConfig()
    parser = argparse.ArgumentParser(
        description="FlowSentry flow probe (inference-first core path)"
    )
    parser.add_argument("network", nargs="?", help="Axelera network name or YAML path")
    parser.add_argument("source", nargs="?", help="Input source, e.g. RTSP URL")
    parser.add_argument("--pipe", dest="pipe_type", default="gst", help="Pipeline type")
    parser.add_argument("--frames", type=int, default=120, help="Max frames to inspect")
    parser.add_argument(
        "--list-flow-networks",
        action="store_true",
        help="List available flow network names and exit",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate env/network only (no deploy/inference)",
    )
    parser.add_argument(
        "--magnitude-threshold",
        type=float,
        default=default_flow_config.mask_magnitude_threshold,
        help="Flow magnitude threshold",
    )
    parser.add_argument(
        "--min-region-area",
        type=int,
        default=default_flow_config.mask_min_region_area,
        help="Minimum flow region area",
    )
    parser.add_argument("--rtsp-latency", type=int, help="RTSP latency in milliseconds")
    parser.add_argument(
        "--low-latency",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable low-latency pipeline mode",
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=7,
        help="Input frame rate for gst pipeline (default: 7, set 0 to keep source FPS)",
    )
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show live window (inference display path); defaults to off when --summary-only is set",
    )
    parser.add_argument(
        "--flow-extract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable flow extraction postprocess (disable for latency isolation)",
    )
    parser.add_argument("--jsonl-out", help="Write per-frame probe records to JSONL")
    parser.add_argument("--summary-json", help="Write probe summary to JSON")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Suppress per-frame stdout and print summary only",
    )
    args = parser.parse_args()
    display_enabled = args.display if args.display is not None else (not args.summary_only)

    if args.list_flow_networks:
        for name in list_flow_network_candidates():
            print(name)
        return

    if not args.network:
        parser.error("network is required unless --list-flow-networks is used")
    if not args.preflight_only and not args.source:
        parser.error("source is required unless --list-flow-networks or --preflight-only is used")

    if args.preflight_only:
        preflight = preflight_axelera_flow_probe(
            network=args.network,
            source=args.source,
            pipe_type=args.pipe_type,
        )
        print(
            "Preflight OK: "
            f"requested_network={preflight.requested_network}, "
            f"resolved_network={preflight.resolved_network}, "
            f"source={preflight.source}, "
            f"pipe={preflight.pipe_type}"
        )
        return

    stream_kwargs: dict[str, int | bool] = {}
    if args.rtsp_latency is not None:
        stream_kwargs["rtsp_latency"] = args.rtsp_latency
    if args.low_latency is not None:
        stream_kwargs["low_latency"] = args.low_latency
    if args.frame_rate > 0:
        stream_kwargs["specified_frame_rate"] = args.frame_rate

    flow_config = FlowConfig(
        mask_magnitude_threshold=args.magnitude_threshold,
        mask_min_region_area=args.min_region_area,
    )
    records, latency = run_axelera_flow_probe_inference_core(
        network=args.network,
        source=args.source,
        pipe_type=args.pipe_type,
        max_frames=args.frames if args.frames > 0 else None,
        flow_config=flow_config,
        stream_kwargs=stream_kwargs,
        display_enabled=display_enabled,
        enable_flow_extract=args.flow_extract,
    )

    summary = summarize_flow_probe_records(records)
    if args.jsonl_out:
        write_flow_probe_jsonl(records, args.jsonl_out)
    if args.summary_json:
        write_flow_probe_summary_json(summary, args.summary_json)

    if not args.summary_only:
        for rec in records:
            print(
                f"[frame={rec.frame_index} stream={rec.stream_id}] "
                f"flow_present={rec.flow_present} consistent={rec.flow_consistent} "
                f"regions={rec.flow_region_count} bbox={rec.flow_bbox}"
            )

    print(
        "Summary: "
        f"frames={summary.frames_processed}, "
        f"frames_with_flow={summary.frames_with_flow}, "
        f"frames_with_consistent_flow={summary.frames_with_consistent_flow}, "
        f"total_flow_regions={summary.total_flow_regions}, "
        f"max_flow_regions={summary.max_flow_regions}, "
        f"first_flow_frame={summary.first_flow_frame_index}, "
        f"streams={summary.stream_ids}"
    )
    print(
        "LatencySummary: "
        f"sampled_frames={latency.sampled_frames}, "
        f"p50_ms={latency.p50_ms}, "
        f"p95_ms={latency.p95_ms}, "
        f"max_ms={latency.max_ms}"
    )


if __name__ == "__main__":
    main()
