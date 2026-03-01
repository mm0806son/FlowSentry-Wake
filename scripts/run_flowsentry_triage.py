#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flowsentry.config import TriageConfig
from flowsentry.runtime.orchestrator import TriageOrchestrator
from flowsentry.types import FrameSignals


def main() -> None:
    cfg = TriageConfig()
    cfg.flow.consistency_frames_threshold = 2
    cfg.runtime.no_motion_reset_frames = 2
    orch = TriageOrchestrator(cfg)

    demo_stream = [
        FrameSignals(False, False, False, None, ()),
        FrameSignals(True, True, True, (10.0, 10.0, 30.0, 30.0), ()),
        FrameSignals(False, True, True, (10.0, 10.0, 30.0, 30.0), ()),
        FrameSignals(False, True, False, (10.0, 10.0, 30.0, 30.0), ((12.0, 12.0, 28.0, 28.0),)),
        FrameSignals(False, False, False, None, ()),
        FrameSignals(False, False, False, None, ()),
    ]

    for idx, signals in enumerate(demo_stream, start=1):
        result = orch.process(signals)
        print(
            f"[{idx}] state={result.state.value} "
            f"flow={result.optical_flow_enabled} yolo={result.yolo_enabled} "
            f"alarm={result.alarm.triggered}:{result.alarm.reason} "
            f"iou={result.match.best_iou}"
        )


if __name__ == "__main__":
    main()
