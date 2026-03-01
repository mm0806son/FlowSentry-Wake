#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="FlowSentry YOLO probe (Axelera runtime)")
    parser.add_argument("network", nargs="?", help="Axelera network name, e.g. yolov8s-coco")
    parser.add_argument("source", nargs="?", help="Input source, e.g. fakevideo or RTSP URL")
    parser.add_argument("--pipe", dest="pipe_type", default="gst", help="Pipeline type (default: gst)")
    parser.add_argument("--frames", type=int, default=30, help="Max frames to inspect")
    parser.add_argument(
        "--list-yolo-networks",
        action="store_true",
        help="List available YOLO network names in current Axelera registry and exit",
    )
    parser.add_argument(
        "--person-conf",
        type=float,
        default=0.25,
        help="Minimum confidence for person detections",
    )
    parser.add_argument(
        "--rtsp-latency",
        type=int,
        help="Optional RTSP latency (ms) passed to Axelera create_inference_stream",
    )
    parser.add_argument("--jsonl-out", help="Write per-frame probe records to JSONL")
    parser.add_argument("--summary-json", help="Write probe summary to JSON")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate Axelera runtime env and resolve network only (no deploy/inference)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Suppress per-frame stdout and print summary only",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Enable overlay visualization (requires cv2)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show window on screen while running (can be used with --save-video).",
    )
    parser.add_argument(
        "--save-video",
        type=str,
        metavar="PATH",
        help="Save video with overlay to PATH (implies --overlay)",
    )
    args = parser.parse_args()

    from flowsentry.runtime import (
        YoloProbeRecord,
        list_yolo_network_candidates,
        preflight_axelera_yolo_probe,
        run_axelera_yolo_probe,
        run_axelera_yolo_probe_with_overlay,
        summarize_probe_records,
        write_probe_jsonl,
        write_probe_summary_json,
    )
    from flowsentry.runtime.adapters import AxeleraYoloBackend

    if args.list_yolo_networks:
        for name in list_yolo_network_candidates():
            print(name)
        return

    if not args.network:
        parser.error("network is required unless --list-yolo-networks is used")
    if not args.preflight_only and not args.source:
        parser.error("source is required unless --list-yolo-networks or --preflight-only is used")

    if args.preflight_only:
        try:
            preflight = preflight_axelera_yolo_probe(
                network=args.network,
                source=args.source,
                pipe_type=args.pipe_type,
            )
        except ValueError as e:
            raise SystemExit(str(e))
        print(
            "Preflight OK: "
            f"requested_network={preflight.requested_network}, "
            f"resolved_network={preflight.resolved_network}, "
            f"source={preflight.source}, "
            f"pipe={preflight.pipe_type}"
        )
        return

    backend = AxeleraYoloBackend(min_confidence=args.person_conf)
    stream_kwargs = {}
    if args.rtsp_latency is not None:
        stream_kwargs["rtsp_latency"] = args.rtsp_latency

    enable_overlay = args.overlay or args.save_video or args.display
    overlay_renderer = None
    display_enabled = args.display or (args.overlay and not args.save_video)

    if enable_overlay:
        from flowsentry.overlay import OverlayConfig, OverlayRenderer
        overlay_renderer = OverlayRenderer(OverlayConfig())

    try:
        if enable_overlay:
            overlay_records = run_axelera_yolo_probe_with_overlay(
                network=args.network,
                source=args.source,
                pipe_type=args.pipe_type,
                max_frames=args.frames,
                yolo_backend=backend,
                stream_kwargs=stream_kwargs,
                overlay_renderer=overlay_renderer,
                video_path=args.save_video,
                display=display_enabled,
            )
            records = [
                YoloProbeRecord(
                    frame_index=r.frame_index,
                    stream_id=r.stream_id,
                    person_count=r.person_count,
                    person_bboxes=r.person_bboxes,
                    primary_person_bbox=r.primary_person_bbox,
                )
                for r in overlay_records
            ]
        else:
            records = run_axelera_yolo_probe(
                network=args.network,
                source=args.source,
                pipe_type=args.pipe_type,
                max_frames=args.frames,
                yolo_backend=backend,
                stream_kwargs=stream_kwargs,
            )
    except ValueError as e:
        raise SystemExit(str(e))
    except FileNotFoundError as e:
        msg = str(e)
        if "$AXELERA_FRAMEWORK" in msg:
            raise SystemExit(
                f"{msg}\nHint: 当前 network 可能是 reference/cascade 配置，路径变量未正确展开。"
                " 请先使用 --list-yolo-networks 选择本机 registry 中的可用 network，"
                " 或直接传入已验证的 YAML 路径。"
            )
        raise
    finally:
        if enable_overlay:
            import cv2
            cv2.destroyAllWindows()

    summary = summarize_probe_records(records)

    if args.jsonl_out:
        write_probe_jsonl(records, args.jsonl_out)
    if args.summary_json:
        write_probe_summary_json(summary, args.summary_json)

    if not args.summary_only:
        for rec in records:
            print(
                f"[frame={rec.frame_index} stream={rec.stream_id}] "
                f"persons={rec.person_count} primary={rec.primary_person_bbox} "
                f"all={rec.person_bboxes}"
            )

    print(
        "Summary: "
        f"frames={summary.frames_processed}, "
        f"frames_with_persons={summary.frames_with_persons}, "
        f"total_persons={summary.total_person_detections}, "
        f"max_persons={summary.max_person_count}, "
        f"first_person_frame={summary.first_person_frame_index}, "
        f"streams={summary.stream_ids}"
    )


if __name__ == "__main__":
    main()
