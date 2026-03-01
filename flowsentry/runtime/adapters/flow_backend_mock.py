from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from flowsentry.runtime.adapters.flow_backend_base import FlowBackendOutput
from flowsentry.types import FlowRegion


@dataclass
class _MockFlowBatch:
    flow_regions: tuple[FlowRegion, ...] = ()
    flow_bbox: tuple[float, ...] | None = None
    flow_present: bool = False
    flow_consistent: bool = False


class MockFlowBackend:
    """Mock flow backend for testing"""

    def __init__(self, batches: list | None = None) -> None:
        self._queue: deque[_MockFlowBatch] = deque()
        if batches:
            for batch in batches:
                if len(batch) == 3:
                    present, consistent, bbox = batch
                    self.push_flow(present=present, consistent=consistent, bbox=bbox)
                else:
                    self._queue.append(_MockFlowBatch())

    def push_flow_regions(self, regions: tuple[FlowRegion, ...]) -> None:
        """Push flow regions to queue"""
        bbox = None
        present = len(regions) > 0
        if present:
            xs, ys, xe, ye = [], [], [], []
            for r in regions:
                box = r.bbox_xyxy
                xs.append(float(box[0]))
                ys.append(float(box[1]))
                xe.append(float(box[2]))
                ye.append(float(box[3]))
            bbox = (min(xs), min(ys), max(xe), max(ye))
        self._queue.append(_MockFlowBatch(
            flow_regions=regions,
            flow_bbox=bbox,
            flow_present=present,
            flow_consistent=present,
        ))

    def push_flow_bbox(self, bbox: tuple[float, float, float, float] | None) -> None:
        """Push flow bbox directly"""
        present = bbox is not None
        regions = ()
        if present:
            x1, y1, x2, y2 = bbox
            area = int((x2 - x1) * (y2 - y1))
            regions = (FlowRegion(bbox_xyxy=np.array(bbox), area=area),)
        self._queue.append(_MockFlowBatch(
            flow_regions=regions,
            flow_bbox=bbox,
            flow_present=present,
            flow_consistent=present,
        ))

    def push_flow(
        self,
        *,
        present: bool = False,
        consistent: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> None:
        """Push flow with explicit parameters"""
        regions = ()
        if present and bbox:
            x1, y1, x2, y2 = bbox
            area = int((x2 - x1) * (y2 - y1))
            regions = (FlowRegion(bbox_xyxy=np.array(bbox), area=area),)
        self._queue.append(_MockFlowBatch(
            flow_regions=regions,
            flow_bbox=bbox if present else None,
            flow_present=present,
            flow_consistent=consistent,
        ))

    def extract(self, frame_result: Any) -> FlowBackendOutput:
        """Extract flow output from frame result"""
        if self._queue:
            batch = self._queue.popleft()
            return FlowBackendOutput(
                flow_regions=batch.flow_regions,
                flow_bbox=batch.flow_bbox,
                flow_present=batch.flow_present,
                flow_consistent=batch.flow_consistent,
            )
        return FlowBackendOutput(
            flow_regions=(),
            flow_bbox=None,
            flow_present=False,
            flow_consistent=False,
        )

    def extract_flow_regions(self, frame_result: Any) -> tuple[FlowRegion, ...]:
        """Extract flow regions only"""
        return self.extract(frame_result).flow_regions

    def reset(self) -> None:
        """Reset internal state"""
        self._queue.clear()
