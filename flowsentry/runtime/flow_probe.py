from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable

from flowsentry.config import FlowConfig
from flowsentry.runtime.adapters import AxeleraFlowBackend, FlowBackend
from flowsentry.runtime.adapters.flow_backend_base import FlowBackendOutput


@dataclass(frozen=True)
class FlowProbeBBoxMeta:
    """Lightweight drawable meta used by native display path."""

    flow_bbox: tuple[float, float, float, float]
    label: str = "Flow"
    color: tuple[int, int, int, int] = (255, 0, 0, 255)

    def draw(self, draw: Any) -> None:
        x1, y1, x2, y2 = [int(round(v)) for v in self.flow_bbox]
        draw.labelled_box((x1, y1), (x2, y2), self.label, self.color)

    def visit(self, callable_: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        callable_(self, *args, **kwargs)


@dataclass(frozen=True)
class FlowProbeRecord:
    frame_index: int
    stream_id: int | None
    flow_present: bool
    flow_consistent: bool
    flow_bbox: tuple[float, float, float, float] | None
    flow_region_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "stream_id": self.stream_id,
            "flow_present": self.flow_present,
            "flow_consistent": self.flow_consistent,
            "flow_bbox": list(self.flow_bbox) if self.flow_bbox is not None else None,
            "flow_region_count": self.flow_region_count,
        }


@dataclass(frozen=True)
class FlowProbeSummary:
    frames_processed: int
    frames_with_flow: int
    frames_with_consistent_flow: int
    total_flow_regions: int
    max_flow_regions: int
    first_flow_frame_index: int | None
    stream_ids: tuple[int | None, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames_processed": self.frames_processed,
            "frames_with_flow": self.frames_with_flow,
            "frames_with_consistent_flow": self.frames_with_consistent_flow,
            "total_flow_regions": self.total_flow_regions,
            "max_flow_regions": self.max_flow_regions,
            "first_flow_frame_index": self.first_flow_frame_index,
            "stream_ids": list(self.stream_ids),
        }


@dataclass(frozen=True)
class FlowProbeLatencySummary:
    sampled_frames: int
    p50_ms: float | None
    p95_ms: float | None
    max_ms: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sampled_frames": self.sampled_frames,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "max_ms": self.max_ms,
        }


@dataclass(frozen=True)
class FlowProbePreflight:
    requested_network: str
    resolved_network: str
    source: str | None
    pipe_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_network": self.requested_network,
            "resolved_network": self.resolved_network,
            "source": self.source,
            "pipe_type": self.pipe_type,
        }


class FlowProbeRunner:
    def __init__(self, *, flow_backend: FlowBackend, max_frames: int | None = None) -> None:
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be >= 1")
        self.flow_backend = flow_backend
        self.max_frames = max_frames

    def run(self, stream: Iterable[Any]) -> Iterable[FlowProbeRecord]:
        for idx, frame_result in enumerate(stream, start=1):
            if self.max_frames is not None and idx > self.max_frames:
                break
            output = self.flow_backend.extract(frame_result)
            yield FlowProbeRecord(
                frame_index=idx,
                stream_id=getattr(frame_result, "stream_id", None),
                flow_present=output.flow_present,
                flow_consistent=output.flow_consistent,
                flow_bbox=output.flow_bbox,
                flow_region_count=len(output.flow_regions),
            )


def _attach_flow_bbox_overlay_meta(
    meta: Any,
    flow_bbox: tuple[float, float, float, float] | None,
    *,
    frame_index: int,
    stream_id: int | None,
) -> Any:
    if flow_bbox is None:
        return meta

    try:
        from axelera.app.meta.base import AxMeta
    except Exception:
        return meta

    overlay_meta = meta
    if overlay_meta is None:
        overlay_meta = AxMeta(image_id=f"flowsentry-flow-probe-{stream_id}-{frame_index}")

    add_instance = getattr(overlay_meta, "add_instance", None)
    if not callable(add_instance):
        return meta

    try:
        delete_instance = getattr(overlay_meta, "delete_instance", None)
        if callable(delete_instance):
            try:
                if "flowsentry_flow_bbox" in overlay_meta:
                    delete_instance("flowsentry_flow_bbox")
            except Exception:
                pass
        add_instance("flowsentry_flow_bbox", FlowProbeBBoxMeta(flow_bbox=flow_bbox))
    except Exception:
        return meta
    return overlay_meta


def list_flow_network_candidates(*, network_yaml_info=None) -> list[str]:
    if network_yaml_info is None:
        from axelera.app import yaml_parser

        network_yaml_info = yaml_parser.get_network_yaml_info()

    names = sorted(set(network_yaml_info.get_all_yaml_names()))
    return [n for n in names if "flow" in n.lower() or "optical" in n.lower()]


def validate_axelera_runtime_env(env: dict[str, str] | None = None) -> None:
    env = env or dict(os.environ)
    framework = env.get("AXELERA_FRAMEWORK", "")
    ld_library_path = env.get("LD_LIBRARY_PATH", "")
    if not framework:
        raise ValueError(
            "AXELERA runtime environment is not activated: missing AXELERA_FRAMEWORK. "
            "Please run 'source venv/bin/activate' before using the flow probe."
        )
    if "runtime" not in ld_library_path and "axelera" not in ld_library_path.lower():
        raise ValueError(
            "AXELERA runtime environment looks incomplete: LD_LIBRARY_PATH does not include runtime libs. "
            "Please run 'source venv/bin/activate' before using the flow probe."
        )


def resolve_flow_network_name(network: str, *, network_yaml_info=None) -> str:
    path = Path(network)
    if path.exists() and path.suffix == ".yaml":
        return str(path)

    if network_yaml_info is None:
        from axelera.app import yaml_parser

        network_yaml_info = yaml_parser.get_network_yaml_info()
    try:
        return network_yaml_info.get_info(network).yaml_name
    except KeyError as e:
        flow_names = list_flow_network_candidates(network_yaml_info=network_yaml_info)
        suggestions = ", ".join(flow_names[:15]) if flow_names else "(none found)"
        raise ValueError(f"{e}. Available flow networks (first 15): {suggestions}") from e


def preflight_axelera_flow_probe(
    *,
    network: str,
    source: str | None = None,
    pipe_type: str = "gst",
    env: dict[str, str] | None = None,
    network_yaml_info=None,
) -> FlowProbePreflight:
    validate_axelera_runtime_env(env=env)
    resolved_network = resolve_flow_network_name(network, network_yaml_info=network_yaml_info)
    return FlowProbePreflight(
        requested_network=network,
        resolved_network=resolved_network,
        source=source,
        pipe_type=pipe_type,
    )


def summarize_flow_probe_records(records: list[FlowProbeRecord]) -> FlowProbeSummary:
    frames_processed = len(records)
    frames_with_flow = sum(1 for r in records if r.flow_present)
    frames_with_consistent_flow = sum(1 for r in records if r.flow_consistent)
    total_flow_regions = sum(r.flow_region_count for r in records)
    max_flow_regions = max((r.flow_region_count for r in records), default=0)
    first_flow = next((r.frame_index for r in records if r.flow_present), None)
    stream_ids = tuple(sorted({r.stream_id for r in records}))
    return FlowProbeSummary(
        frames_processed=frames_processed,
        frames_with_flow=frames_with_flow,
        frames_with_consistent_flow=frames_with_consistent_flow,
        total_flow_regions=total_flow_regions,
        max_flow_regions=max_flow_regions,
        first_flow_frame_index=first_flow,
        stream_ids=stream_ids,
    )


def write_flow_probe_jsonl(records: list[FlowProbeRecord], path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    return out_path


def write_flow_probe_summary_json(summary: FlowProbeSummary, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return out_path


def summarize_frame_age_samples(frame_age_samples_s: list[float]) -> FlowProbeLatencySummary:
    if not frame_age_samples_s:
        return FlowProbeLatencySummary(sampled_frames=0, p50_ms=None, p95_ms=None, max_ms=None)

    sorted_samples = sorted(frame_age_samples_s)
    p50 = _percentile_linear(sorted_samples, 0.50) * 1000.0
    p95 = _percentile_linear(sorted_samples, 0.95) * 1000.0
    max_v = sorted_samples[-1] * 1000.0
    return FlowProbeLatencySummary(
        sampled_frames=len(sorted_samples),
        p50_ms=round(p50, 2),
        p95_ms=round(p95, 2),
        max_ms=round(max_v, 2),
    )


def run_axelera_flow_probe(
    *,
    network: str,
    source: str,
    pipe_type: str = "gst",
    max_frames: int | None = None,
    flow_backend: FlowBackend | None = None,
    flow_config: FlowConfig | None = None,
    stream_factory: Callable[..., Any] | None = None,
    stream_kwargs: dict[str, Any] | None = None,
) -> list[FlowProbeRecord]:
    backend = flow_backend or AxeleraFlowBackend(config=flow_config)
    resolved_network = network
    stream_kwargs = dict(stream_kwargs or {})
    if stream_factory is None:
        from axelera.app import yaml_parser

        preflight = preflight_axelera_flow_probe(
            network=network,
            source=source,
            pipe_type=pipe_type,
            network_yaml_info=yaml_parser.get_network_yaml_info(),
        )
        resolved_network = preflight.resolved_network
        stream = _create_inference_stream_like_inference(
            network=resolved_network,
            source=source,
            pipe_type=pipe_type,
            stream_kwargs=stream_kwargs,
        )
    else:
        stream = stream_factory(
            network=resolved_network,
            sources=[source],
            pipe_type=pipe_type,
            **stream_kwargs,
        )

    runner = FlowProbeRunner(flow_backend=backend, max_frames=max_frames)
    try:
        return list(runner.run(stream))
    finally:
        stop = getattr(stream, "stop", None)
        if callable(stop):
            stop()


def run_axelera_flow_probe_inference_core(
    *,
    network: str,
    source: str,
    pipe_type: str = "gst",
    max_frames: int | None = None,
    flow_backend: FlowBackend | None = None,
    flow_config: FlowConfig | None = None,
    stream_factory: Callable[..., Any] | None = None,
    stream_kwargs: dict[str, Any] | None = None,
    display_enabled: bool = True,
    enable_flow_extract: bool = True,
    window_size: tuple[int, int] = (1280, 720),
) -> tuple[list[FlowProbeRecord], FlowProbeLatencySummary]:
    backend = flow_backend or AxeleraFlowBackend(config=flow_config)
    resolved_network = network
    stream_kwargs = dict(stream_kwargs or {})
    if stream_factory is None:
        from axelera.app import yaml_parser

        preflight = preflight_axelera_flow_probe(
            network=network,
            source=source,
            pipe_type=pipe_type,
            network_yaml_info=yaml_parser.get_network_yaml_info(),
        )
        resolved_network = preflight.resolved_network
        stream = _create_inference_stream_like_inference(
            network=resolved_network,
            source=source,
            pipe_type=pipe_type,
            stream_kwargs=stream_kwargs,
        )
    else:
        stream = stream_factory(
            network=resolved_network,
            sources=[source],
            pipe_type=pipe_type,
            **stream_kwargs,
        )

    records: list[FlowProbeRecord] = []
    frame_ages_s: list[float] = []

    def _consume_results(wnd=None):
        frame_idx = 0
        for event in stream.with_events():
            frame_result = getattr(event, "result", None)
            if frame_result is None:
                if wnd is not None and wnd.is_closed:
                    break
                continue

            frame_idx += 1
            if max_frames is not None and frame_idx > max_frames:
                break

            image = getattr(frame_result, "image", None)
            meta = getattr(frame_result, "meta", None)
            if wnd is not None and image is not None and not wnd.is_closed:
                wnd.show(image, meta, frame_result.stream_id)
            if wnd is not None and wnd.is_closed:
                break

            if enable_flow_extract:
                output = backend.extract(frame_result)
            else:
                output = FlowBackendOutput(
                    flow_regions=(),
                    flow_bbox=None,
                    flow_present=False,
                    flow_consistent=False,
                )
            records.append(
                FlowProbeRecord(
                    frame_index=frame_idx,
                    stream_id=getattr(frame_result, "stream_id", None),
                    flow_present=output.flow_present,
                    flow_consistent=output.flow_consistent,
                    flow_bbox=output.flow_bbox,
                    flow_region_count=len(output.flow_regions),
                )
            )

            age_s = _frame_age_seconds(frame_result)
            if age_s is not None:
                frame_ages_s.append(age_s)

    try:
        if display_enabled:
            from axelera.app import display

            with display.App(
                renderer="auto",
                opengl=stream.hardware_caps.opengl,
                buffering=not stream.is_single_image(),
            ) as app:
                wnd = app.create_window("FlowSentry Flow Probe (Inference Core)", size=window_size)
                app.start_thread(lambda: _consume_results(wnd), name="FlowProbeInferenceCoreThread")
                app.run(interval=1 / 10)
        else:
            _consume_results()
    finally:
        stop = getattr(stream, "stop", None)
        if callable(stop):
            stop()

    return records, summarize_frame_age_samples(frame_ages_s)


@dataclass(frozen=True)
class FlowProbeOverlayRecord:
    frame_index: int
    stream_id: int | None
    flow_present: bool
    flow_consistent: bool
    flow_bbox: tuple[float, float, float, float] | None
    flow_region_count: int
    frame_age_s: float | None = None
    annotated_frame: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "stream_id": self.stream_id,
            "flow_present": self.flow_present,
            "flow_consistent": self.flow_consistent,
            "flow_bbox": list(self.flow_bbox) if self.flow_bbox is not None else None,
            "flow_region_count": self.flow_region_count,
            "frame_age_s": self.frame_age_s,
        }


def run_axelera_flow_probe_with_display(
    *,
    network: str,
    source: str,
    pipe_type: str = "gst",
    max_frames: int | None = None,
    flow_backend: FlowBackend | None = None,
    flow_config: FlowConfig | None = None,
    stream_kwargs: dict[str, Any] | None = None,
    max_frame_age_s: float | None = None,
    window_size: tuple[int, int] = (1280, 720),
) -> list[FlowProbeOverlayRecord]:
    from axelera.app import display

    backend = flow_backend or AxeleraFlowBackend(config=flow_config)
    stream_kwargs = dict(stream_kwargs or {})
    stream = _create_inference_stream_like_inference(
        network=network,
        source=source,
        pipe_type=pipe_type,
        stream_kwargs=stream_kwargs,
    )
    records: list[FlowProbeOverlayRecord] = []

    try:
        with display.App(
            renderer="auto",
            opengl=stream.hardware_caps.opengl,
            buffering=not stream.is_single_image(),
        ) as app:
            wnd = app.create_window("FlowSentry Flow Probe", size=window_size)

            def process_frames():
                frame_count = 0
                for event in stream.with_events():
                    frame_result = getattr(event, "result", None)
                    if frame_result is None:
                        if wnd.is_closed:
                            break
                        continue
                    if max_frames is not None and frame_count >= max_frames:
                        break

                    frame_count += 1
                    if _is_stale_frame(frame_result, max_frame_age_s):
                        continue

                    output = backend.extract(frame_result)
                    image = getattr(frame_result, "image", None)
                    meta = getattr(frame_result, "meta", None)
                    stream_id = getattr(frame_result, "stream_id", None)
                    display_meta = _attach_flow_bbox_overlay_meta(
                        meta,
                        output.flow_bbox,
                        frame_index=frame_count,
                        stream_id=stream_id,
                    )

                    if image is not None and not wnd.is_closed:
                        wnd.show(image, display_meta, stream_id)
                    if wnd.is_closed:
                        break

                    records.append(
                        FlowProbeOverlayRecord(
                            frame_index=frame_count,
                            stream_id=stream_id,
                            flow_present=output.flow_present,
                            flow_consistent=output.flow_consistent,
                            flow_bbox=output.flow_bbox,
                            flow_region_count=len(output.flow_regions),
                            frame_age_s=_frame_age_seconds(frame_result),
                            annotated_frame=None,
                        )
                    )

            app.start_thread(process_frames, name="FlowProbeThread")
            app.run(interval=1 / 10)

    finally:
        stop = getattr(stream, "stop", None)
        if callable(stop):
            stop()
    return records


def run_axelera_flow_probe_with_overlay(
    *,
    network: str,
    source: str,
    pipe_type: str = "gst",
    max_frames: int | None = None,
    flow_backend: FlowBackend | None = None,
    flow_config: FlowConfig | None = None,
    stream_factory: Callable[..., Any] | None = None,
    stream_kwargs: dict[str, Any] | None = None,
    overlay_renderer: Any | None = None,
    video_path: str | Path | None = None,
    display: bool = False,
    use_native_display: bool = False,
    max_frame_age_s: float | None = None,
) -> list[FlowProbeOverlayRecord]:
    if use_native_display and display:
        return run_axelera_flow_probe_with_display(
            network=network,
            source=source,
            pipe_type=pipe_type,
            max_frames=max_frames,
            flow_backend=flow_backend,
            flow_config=flow_config,
            stream_kwargs=stream_kwargs,
            max_frame_age_s=max_frame_age_s,
        )

    import numpy as np

    backend = flow_backend or AxeleraFlowBackend(config=flow_config)
    resolved_network = network
    stream_kwargs = dict(stream_kwargs or {})
    if stream_factory is None:
        from axelera.app import yaml_parser

        preflight = preflight_axelera_flow_probe(
            network=network,
            source=source,
            pipe_type=pipe_type,
            network_yaml_info=yaml_parser.get_network_yaml_info(),
        )
        resolved_network = preflight.resolved_network
        stream = _create_inference_stream_like_inference(
            network=resolved_network,
            source=source,
            pipe_type=pipe_type,
            stream_kwargs=stream_kwargs,
        )
    else:
        stream = stream_factory(
            network=resolved_network,
            sources=[source],
            pipe_type=pipe_type,
            **stream_kwargs,
        )

    video_path_obj = Path(video_path) if video_path else None
    if video_path_obj is not None:
        video_path_obj.parent.mkdir(parents=True, exist_ok=True)

    video_writer_local = None
    records: list[FlowProbeOverlayRecord] = []
    try:
        for idx, frame_result in enumerate(stream, start=1):
            if max_frames is not None and idx > max_frames:
                break

            if _is_stale_frame(frame_result, max_frame_age_s):
                continue
            output = backend.extract(frame_result)

            annotated_frame = None
            if overlay_renderer is not None or video_path_obj is not None or display:
                image = getattr(frame_result, "image", None)
                if image is not None:
                    if hasattr(image, "asarray"):
                        frame = np.asarray(image.asarray())
                    else:
                        frame = np.asarray(image)
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    if overlay_renderer is not None:
                        from flowsentry.types import FrameSignals

                        signals = FrameSignals(
                            frame_diff_triggered=False,
                            flow_present=output.flow_present,
                            flow_consistent=output.flow_consistent,
                            flow_bbox=output.flow_bbox,
                            person_bboxes=(),
                        )
                        from flowsentry.runtime.orchestrator import TriageOrchestratorOutput
                        from flowsentry.fsm import TriageState
                        from flowsentry.types import AlarmDecision
                        from flowsentry.fusion import IoUMatchResult

                        dummy_result = TriageOrchestratorOutput(
                            state=TriageState.FLOW_ACTIVE
                            if output.flow_present
                            else TriageState.STANDBY,
                            optical_flow_enabled=True,
                            yolo_enabled=False,
                            consistency_count=1 if output.flow_consistent else 0,
                            flow_threshold_reached=output.flow_consistent,
                            match=IoUMatchResult(
                                matched=False,
                                best_iou=None,
                                best_person_bbox=None,
                                threshold=0.3,
                            ),
                            alarm=AlarmDecision(False, reason="flow_probe_only"),
                        )
                        annotated_frame = overlay_renderer.render(frame, signals, dummy_result)
                    else:
                        annotated_frame = frame

            if (
                video_path_obj is not None
                and annotated_frame is not None
                and video_writer_local is None
            ):
                import cv2

                h, w = annotated_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer_local = cv2.VideoWriter(str(video_path_obj), fourcc, 30.0, (w, h))

            if video_writer_local is not None and annotated_frame is not None:
                if annotated_frame.ndim == 3 and annotated_frame.shape[2] == 3:
                    bgr = annotated_frame[:, :, ::-1].copy()
                else:
                    bgr = annotated_frame
                video_writer_local.write(bgr)

            if display and annotated_frame is not None:
                import cv2

                if annotated_frame.ndim == 3 and annotated_frame.shape[2] == 3:
                    display_frame = annotated_frame[:, :, ::-1].copy()
                else:
                    display_frame = annotated_frame
                cv2.imshow("FlowSentry Flow Probe", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            records.append(
                FlowProbeOverlayRecord(
                    frame_index=idx,
                    stream_id=getattr(frame_result, "stream_id", None),
                    flow_present=output.flow_present,
                    flow_consistent=output.flow_consistent,
                    flow_bbox=output.flow_bbox,
                    flow_region_count=len(output.flow_regions),
                    frame_age_s=_frame_age_seconds(frame_result),
                    annotated_frame=annotated_frame,
                )
            )
    finally:
        if video_writer_local is not None:
            video_writer_local.release()
        stop = getattr(stream, "stop", None)
        if callable(stop):
            stop()

    return records


def _frame_age_seconds(frame_result: Any, *, now_s: float | None = None) -> float | None:
    src_timestamp = getattr(frame_result, "src_timestamp", None)
    if src_timestamp is None:
        return None
    try:
        src_s = float(src_timestamp)
    except (TypeError, ValueError):
        return None
    if src_s <= 0:
        return None

    now = time.time() if now_s is None else now_s
    age = now - src_s
    return age if age >= 0 else 0.0


def _is_stale_frame(frame_result: Any, max_frame_age_s: float | None) -> bool:
    if max_frame_age_s is None or max_frame_age_s <= 0:
        return False
    age = _frame_age_seconds(frame_result)
    return age is not None and age > max_frame_age_s


def _percentile_linear(sorted_values: list[float], ratio: float) -> float:
    if not sorted_values:
        raise ValueError("sorted_values must not be empty")
    if ratio <= 0:
        return sorted_values[0]
    if ratio >= 1:
        return sorted_values[-1]

    n = len(sorted_values)
    pos = (n - 1) * ratio
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _create_inference_stream_like_inference(
    *,
    network: str,
    source: str,
    pipe_type: str,
    stream_kwargs: dict[str, Any],
):
    from axelera.app import config, stream as ax_stream

    kwargs = dict(stream_kwargs)
    pipeline_config = config.PipelineConfig(
        network=network,
        sources=[source],
        pipe_type=pipe_type,
    )
    pipeline_config.update_from_kwargs(kwargs)

    return ax_stream.create_inference_stream(
        config.SystemConfig(),
        config.InferenceStreamConfig(),
        pipeline_config,
        config.LoggingConfig(),
        config.DeployConfig(),
        **kwargs,
    )
