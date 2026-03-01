from __future__ import annotations

from collections import deque
from typing import Any, Iterable

from flowsentry.runtime.adapters.yolo_backend_base import YoloBackendOutput
from flowsentry.types import BBox, Detection


class MockYoloBackend:
    """Queue-based mock backend to drive orchestrator tests and demos."""

    def __init__(self, batches: Iterable[tuple[BBox, ...]] | None = None) -> None:
        self._queue: deque[tuple[BBox, ...]] = deque(batches or [])

    def push_person_bboxes(self, person_bboxes: tuple[BBox, ...]) -> None:
        self._queue.append(person_bboxes)

    def extract(self, _frame_result: Any = None) -> YoloBackendOutput:
        person_bboxes = self._queue.popleft() if self._queue else ()
        detections = tuple(
            Detection(
                bbox_xyxy=b,
                class_id=0,
                class_name="person",
                confidence=1.0,
            )
            for b in person_bboxes
        )
        return YoloBackendOutput(
            detections=detections,
            person_bboxes=person_bboxes,
            all_detections=detections,
        )

    def extract_person_bboxes(self, frame_result: Any = None) -> tuple[BBox, ...]:
        return self.extract(frame_result).person_bboxes
