from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable

from flowsentry.runtime.adapters import AxeleraYoloBackend, YoloBackend
from flowsentry.vision.yolo_filter import select_primary_person_bbox


@dataclass(frozen=True)
class YoloProbeRecord:
    frame_index: int
    stream_id: int | None
    person_count: int
    person_bboxes: tuple[tuple[float, float, float, float], ...]
    primary_person_bbox: tuple[float, float, float, float] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "stream_id": self.stream_id,
            "person_count": self.person_count,
            "person_bboxes": [list(b) for b in self.person_bboxes],
            "primary_person_bbox": (
                list(self.primary_person_bbox) if self.primary_person_bbox is not None else None
            ),
        }


@dataclass(frozen=True)
class YoloProbeSummary:
    frames_processed: int
    frames_with_persons: int
    total_person_detections: int
    max_person_count: int
    first_person_frame_index: int | None
    stream_ids: tuple[int | None, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames_processed": self.frames_processed,
            "frames_with_persons": self.frames_with_persons,
            "total_person_detections": self.total_person_detections,
            "max_person_count": self.max_person_count,
            "first_person_frame_index": self.first_person_frame_index,
            "stream_ids": list(self.stream_ids),
        }


@dataclass(frozen=True)
class YoloProbePreflight:
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


class YoloProbeRunner:
    def __init__(self, *, yolo_backend: YoloBackend, max_frames: int | None = None) -> None:
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be >= 1")
        self.yolo_backend = yolo_backend
        self.max_frames = max_frames

    def run(self, stream: Iterable[Any]) -> Iterable[YoloProbeRecord]:
        for idx, frame_result in enumerate(stream, start=1):
            if self.max_frames is not None and idx > self.max_frames:
                break
            person_bboxes = self.yolo_backend.extract_person_bboxes(frame_result)
            yield YoloProbeRecord(
                frame_index=idx,
                stream_id=getattr(frame_result, "stream_id", None),
                person_count=len(person_bboxes),
                person_bboxes=person_bboxes,
                primary_person_bbox=select_primary_person_bbox(person_bboxes),
            )


def list_yolo_network_candidates(*, network_yaml_info=None) -> list[str]:
    if network_yaml_info is None:
        from axelera.app import yaml_parser  # lazy import

        network_yaml_info = yaml_parser.get_network_yaml_info()

    names = sorted(set(network_yaml_info.get_all_yaml_names()))
    return [n for n in names if "yolo" in n.lower()]


def validate_axelera_runtime_env(env: dict[str, str] | None = None) -> None:
    env = env or dict(os.environ)
    framework = env.get("AXELERA_FRAMEWORK", "")
    ld_library_path = env.get("LD_LIBRARY_PATH", "")
    if not framework:
        raise ValueError(
            "AXELERA runtime environment is not activated: missing AXELERA_FRAMEWORK. "
            "Please run 'source venv/bin/activate' before using the YOLO probe."
        )
    if "runtime" not in ld_library_path and "axelera" not in ld_library_path.lower():
        raise ValueError(
            "AXELERA runtime environment looks incomplete: LD_LIBRARY_PATH does not include runtime libs. "
            "Please run 'source venv/bin/activate' before using the YOLO probe."
        )


def resolve_probe_network_name(network: str, *, network_yaml_info=None) -> str:
    path = Path(network)
    if path.exists() and path.suffix == ".yaml":
        return str(path)

    if network_yaml_info is None:
        from axelera.app import yaml_parser  # lazy import

        network_yaml_info = yaml_parser.get_network_yaml_info()
    try:
        return network_yaml_info.get_info(network).yaml_name
    except KeyError as e:
        yolo_names = list_yolo_network_candidates(network_yaml_info=network_yaml_info)
        suggestions = ", ".join(yolo_names[:15]) if yolo_names else "(none found)"
        raise ValueError(
            f"{e}. Available YOLO networks (first 15): {suggestions}"
        ) from e


def preflight_axelera_yolo_probe(
    *,
    network: str,
    source: str | None = None,
    pipe_type: str = "gst",
    env: dict[str, str] | None = None,
    network_yaml_info=None,
) -> YoloProbePreflight:
    validate_axelera_runtime_env(env=env)
    resolved_network = resolve_probe_network_name(network, network_yaml_info=network_yaml_info)
    return YoloProbePreflight(
        requested_network=network,
        resolved_network=resolved_network,
        source=source,
        pipe_type=pipe_type,
    )


def summarize_probe_records(records: list[YoloProbeRecord]) -> YoloProbeSummary:
    frames_processed = len(records)
    frames_with_persons = sum(1 for r in records if r.person_count > 0)
    total_person_detections = sum(r.person_count for r in records)
    max_person_count = max((r.person_count for r in records), default=0)
    first_person = next((r.frame_index for r in records if r.person_count > 0), None)
    stream_ids = tuple(sorted({r.stream_id for r in records if r.stream_id is not None}))
    return YoloProbeSummary(
        frames_processed=frames_processed,
        frames_with_persons=frames_with_persons,
        total_person_detections=total_person_detections,
        max_person_count=max_person_count,
        first_person_frame_index=first_person,
        stream_ids=stream_ids,
    )


def write_probe_jsonl(records: list[YoloProbeRecord], path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    return out_path


def write_probe_summary_json(summary: YoloProbeSummary, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_path


def run_axelera_yolo_probe(
    *,
    network: str,
    source: str,
    pipe_type: str = "gst",
    max_frames: int | None = None,
    yolo_backend: YoloBackend | None = None,
    stream_factory: Callable[..., Any] | None = None,
    stream_kwargs: dict[str, Any] | None = None,
) -> list[YoloProbeRecord]:
    backend = yolo_backend or AxeleraYoloBackend()
    resolved_network = network
    stream_kwargs = dict(stream_kwargs or {})
    if stream_factory is None:
        from axelera.app import yaml_parser  # lazy import for testability
        from axelera.app.stream import create_inference_stream  # lazy import for testability

        preflight = preflight_axelera_yolo_probe(
            network=network,
            source=source,
            pipe_type=pipe_type,
            network_yaml_info=yaml_parser.get_network_yaml_info(),
        )
        resolved_network = preflight.resolved_network
        stream_factory = create_inference_stream

    stream = stream_factory(
        network=resolved_network,
        sources=[source],
        pipe_type=pipe_type,
        **stream_kwargs,
    )

    runner = YoloProbeRunner(yolo_backend=backend, max_frames=max_frames)
    try:
        return list(runner.run(stream))
    finally:
        stop = getattr(stream, "stop", None)
        if callable(stop):
            stop()


@dataclass(frozen=True)
class YoloProbeOverlayRecord:
    frame_index: int
    stream_id: int | None
    person_count: int
    person_bboxes: tuple[tuple[float, float, float, float], ...]
    primary_person_bbox: tuple[float, float, float, float] | None
    annotated_frame: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "frame_index": self.frame_index,
            "stream_id": self.stream_id,
            "person_count": self.person_count,
            "person_bboxes": [list(b) for b in self.person_bboxes],
            "primary_person_bbox": (
                list(self.primary_person_bbox) if self.primary_person_bbox is not None else None
            ),
        }
        return d


def run_axelera_yolo_probe_with_overlay(
    *,
    network: str,
    source: str,
    pipe_type: str = "gst",
    max_frames: int | None = None,
    yolo_backend: YoloBackend | None = None,
    stream_factory: Callable[..., Any] | None = None,
    stream_kwargs: dict[str, Any] | None = None,
    overlay_renderer: Any | None = None,
    video_path: str | Path | None = None,
    video_writer: Any | None = None,
    display: bool = False,
) -> list[YoloProbeOverlayRecord]:
    import numpy as np

    backend = yolo_backend or AxeleraYoloBackend()
    resolved_network = network
    stream_kwargs = dict(stream_kwargs or {})
    if stream_factory is None:
        from axelera.app import yaml_parser
        from axelera.app.stream import create_inference_stream

        preflight = preflight_axelera_yolo_probe(
            network=network,
            source=source,
            pipe_type=pipe_type,
            network_yaml_info=yaml_parser.get_network_yaml_info(),
        )
        resolved_network = preflight.resolved_network
        stream_factory = create_inference_stream

    stream = stream_factory(
        network=resolved_network,
        sources=[source],
        pipe_type=pipe_type,
        **stream_kwargs,
    )

    video_path_obj = Path(video_path) if video_path else None
    if video_path_obj is not None:
        video_path_obj.parent.mkdir(parents=True, exist_ok=True)

    video_writer_local = video_writer
    owns_video_writer = False
    records: list[YoloProbeOverlayRecord] = []
    try:
        for idx, frame_result in enumerate(stream, start=1):
            if max_frames is not None and idx > max_frames:
                break

            person_bboxes = backend.extract_person_bboxes(frame_result)
            primary_bbox = select_primary_person_bbox(person_bboxes)

            annotated_frame = None
            if (
                overlay_renderer is not None
                or video_writer is not None
                or video_path_obj is not None
                or display
            ):
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
                            flow_present=False,
                            flow_consistent=False,
                            flow_bbox=None,
                            person_bboxes=person_bboxes,
                        )
                        from flowsentry.runtime.orchestrator import TriageOrchestratorOutput
                        from flowsentry.fsm import TriageState
                        from flowsentry.types import AlarmDecision
                        from flowsentry.fusion import IoUMatchResult
                        dummy_result = TriageOrchestratorOutput(
                            state=TriageState.STANDBY,
                            optical_flow_enabled=False,
                            yolo_enabled=True,
                            consistency_count=0,
                            flow_threshold_reached=False,
                            match=IoUMatchResult(
                                matched=False,
                                best_iou=None,
                                best_person_bbox=None,
                                threshold=0.3,
                            ),
                            alarm=AlarmDecision(False, reason="yolo_probe_only"),
                        )
                        annotated_frame = overlay_renderer.render(frame, signals, dummy_result)
                    else:
                        annotated_frame = frame

            if (
                video_writer_local is None
                and video_path_obj is not None
                and annotated_frame is not None
            ):
                import cv2

                h, w = annotated_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer_local = cv2.VideoWriter(str(video_path_obj), fourcc, 30.0, (w, h))
                owns_video_writer = True

            if video_writer_local is not None and annotated_frame is not None:
                bgr = _to_bgr_for_cv(annotated_frame)
                video_writer_local.write(bgr)

            if display and annotated_frame is not None:
                import cv2
                cv2.imshow("FlowSentry YOLO Probe", _to_bgr_for_cv(annotated_frame))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            records.append(YoloProbeOverlayRecord(
                frame_index=idx,
                stream_id=getattr(frame_result, "stream_id", None),
                person_count=len(person_bboxes),
                person_bboxes=person_bboxes,
                primary_person_bbox=primary_bbox,
                annotated_frame=annotated_frame,
            ))
    finally:
        if owns_video_writer and video_writer_local is not None:
            video_writer_local.release()
        stop = getattr(stream, "stop", None)
        if callable(stop):
            stop()

    return records


def _to_bgr_for_cv(frame: Any) -> Any:
    if getattr(frame, "ndim", 0) == 3 and frame.shape[2] == 3:
        return frame[:, :, ::-1].copy()
    return frame
