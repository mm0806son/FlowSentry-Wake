from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from flowsentry.types import BBox, Detection


@dataclass(frozen=True)
class YoloBackendOutput:
    detections: tuple[Detection, ...]
    person_bboxes: tuple[BBox, ...]
    all_detections: tuple[Detection, ...] = ()


class YoloBackend(Protocol):
    def extract(self, frame_result: Any) -> YoloBackendOutput:
        ...

    def extract_person_bboxes(self, frame_result: Any) -> tuple[BBox, ...]:
        ...
