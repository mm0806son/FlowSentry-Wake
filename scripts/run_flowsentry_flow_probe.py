#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    from flowsentry.config import FlowConfig

    default_flow_config = FlowConfig()
    parser = argparse.ArgumentParser(
        description="FlowSentry flow probe (Axelera EdgeFlownet runtime)"
    )
    parser.add_argument(
        "network", nargs="?", help="Axelera network name, e.g. edgeflownet-opticalflow"
    )
    parser.add_argument("source", nargs="?", help="Input source, e.g. fakevideo or RTSP URL")
    parser.add_argument(
        "--pipe", dest="pipe_type", default="gst", help="Pipeline type (default: gst)"
    )
    parser.add_argument("--frames", type=int, default=30, help="Max frames to inspect")
    parser.add_argument(
        "--list-flow-networks",
        action="store_true",
        help="List available flow network names in current Axelera registry and exit",
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
        "--rtsp-latency",
        type=int,
        default=100,
        help="Optional RTSP latency (ms) passed to Axelera create_inference_stream",
    )
    parser.add_argument(
        "--low-latency",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable low-latency pipeline mode (disabled by default unless explicitly set).",
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=7,
        help=(
            "Input frame rate for gst pipeline (default: 7). "
            "Set 0 to keep source FPS."
        ),
    )
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Show live window in inference-core path when --overlay is not used. "
            "Default disabled for probe/latency runs."
        ),
    )
    parser.add_argument(
        "--flow-extract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable flow extraction postprocess (disable for latency isolation)",
    )
    parser.add_argument(
        "--debug-flow-tensor",
        action="store_true",
        help=(
            "Print per-frame flow tensor extraction diagnostics "
            "(source/shape/magnitude stats)."
        ),
    )
    parser.add_argument(
        "--use-dmabuf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable DMA buffer transfers. Default enabled for lower latency; "
            "use --no-use-dmabuf if runtime stability issues are observed."
        ),
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
        help=(
            "Enable overlay visualization. Live view uses native display by default; "
            "cv2 path is used for --save-video or --no-native-display."
        ),
    )
    parser.add_argument(
        "--native-display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use native Axelera display path for live --overlay view "
            "(recommended on Orange Pi for lower CPU load)."
        ),
    )
    parser.add_argument(
        "--save-video",
        type=str,
        metavar="PATH",
        help="Save video with overlay to PATH (implies --overlay)",
    )
    args = parser.parse_args()

    if args.use_dmabuf:
        if os.environ.get("AXELERA_USE_DMABUF") == "0":
            os.environ.pop("AXELERA_USE_DMABUF")
    else:
        os.environ["AXELERA_USE_DMABUF"] = "0"

    from flowsentry.runtime import (
        FlowProbeRecord,
        list_flow_network_candidates,
        preflight_axelera_flow_probe,
        run_axelera_flow_probe_inference_core,
        run_axelera_flow_probe_with_overlay,
        summarize_frame_age_samples,
        summarize_flow_probe_records,
        write_flow_probe_jsonl,
        write_flow_probe_summary_json,
    )
    from flowsentry.runtime.adapters import AxeleraFlowBackend

    if args.list_flow_networks:
        for name in list_flow_network_candidates():
            print(name)
        return

    if not args.network:
        parser.error("network is required unless --list-flow-networks is used")
    if not args.preflight_only and not args.source:
        parser.error("source is required unless --list-flow-networks or --preflight-only is used")

    if args.preflight_only:
        try:
            preflight = preflight_axelera_flow_probe(
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

    flow_config = FlowConfig(
        mask_magnitude_threshold=args.magnitude_threshold,
        mask_min_region_area=args.min_region_area,
    )
    debug_flow_counter = {"frame_index": 0}
    debug_callback = None
    if args.debug_flow_tensor:

        def _debug_printer(info: dict[str, object]) -> None:
            debug_flow_counter["frame_index"] += 1
            idx = debug_flow_counter["frame_index"]
            candidates = tuple(info.get("candidates") or ())
            preview = ", ".join(str(c) for c in candidates[:4]) if candidates else "-"
            if len(candidates) > 4:
                preview = f"{preview}, ..."
            print(
                f"[flow_debug frame={idx}] "
                f"found={info.get('flow_found')} "
                f"source={info.get('source')} "
                f"mode={info.get('decode_mode')} "
                f"raw_shape={info.get('raw_shape')} "
                f"flow_shape={info.get('flow_shape')} "
                f"mag_p50={info.get('magnitude_p50')} "
                f"mag_p95={info.get('magnitude_p95')} "
                f"mag_max={info.get('magnitude_max')} "
                f"candidates={info.get('candidate_count')} "
                f"[{preview}]"
            )

        debug_callback = _debug_printer

    backend = AxeleraFlowBackend(config=flow_config, debug_callback=debug_callback)
    enable_overlay = args.overlay or args.save_video
    stream_kwargs: dict[str, int | bool] = {}

    if args.rtsp_latency is not None:
        stream_kwargs["rtsp_latency"] = args.rtsp_latency

    low_latency = args.low_latency
    if low_latency is not None:
        stream_kwargs["low_latency"] = low_latency

    if args.frame_rate > 0:
        stream_kwargs["specified_frame_rate"] = args.frame_rate

    overlay_renderer = None

    if enable_overlay:
        from flowsentry.overlay import OverlayConfig, OverlayRenderer

        overlay_renderer = OverlayRenderer(OverlayConfig())

    try:
        if enable_overlay:
            overlay_records = run_axelera_flow_probe_with_overlay(
                network=args.network,
                source=args.source,
                pipe_type=args.pipe_type,
                max_frames=args.frames,
                flow_backend=backend,
                stream_kwargs=stream_kwargs,
                overlay_renderer=overlay_renderer,
                video_path=args.save_video,
                display=args.overlay and not args.save_video,
                use_native_display=args.native_display,
            )
            records = [
                FlowProbeRecord(
                    frame_index=r.frame_index,
                    stream_id=r.stream_id,
                    flow_present=r.flow_present,
                    flow_consistent=r.flow_consistent,
                    flow_bbox=r.flow_bbox,
                    flow_region_count=r.flow_region_count,
                )
                for r in overlay_records
            ]
            latency = summarize_frame_age_samples(
                [r.frame_age_s for r in overlay_records if r.frame_age_s is not None]
            )
        else:
            records, latency = run_axelera_flow_probe_inference_core(
                network=args.network,
                source=args.source,
                pipe_type=args.pipe_type,
                max_frames=args.frames if args.frames > 0 else None,
                flow_backend=backend,
                stream_kwargs=stream_kwargs,
                display_enabled=args.display,
                enable_flow_extract=args.flow_extract,
            )
    except ValueError as e:
        raise SystemExit(str(e))
    except FileNotFoundError as e:
        msg = str(e)
        if "$AXELERA_FRAMEWORK" in msg:
            raise SystemExit(
                f"{msg}\nHint: 当前 network 可能是 reference/cascade 配置，路径变量未正确展开。"
                " 请先使用 --list-flow-networks 选择本机 registry 中的可用 network，"
                " 或直接传入已验证的 YAML 路径。"
            )
        raise
    finally:
        if enable_overlay and args.save_video:
            import cv2

            cv2.destroyAllWindows()

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
