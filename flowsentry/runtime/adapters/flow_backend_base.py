from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from flowsentry.types import BBox, FlowRegion


@dataclass(frozen=True)
class FlowBackendOutput:
    flow_regions: tuple[FlowRegion, ...]
    flow_bbox: BBox | None
    flow_present: bool
    flow_consistent: bool


class FlowBackend(Protocol):
    def extract(self, frame_result: Any) -> FlowBackendOutput:
        ...

    def extract_flow_regions(self, frame_result: Any) -> tuple[FlowRegion, ...]:
        ...

    def reset(self) -> None:
        ...
