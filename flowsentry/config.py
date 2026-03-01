from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class FrameDiffConfig:
    min_area: int = 400
    pixel_change_ratio_threshold: float = 0.01


@dataclass(slots=True)
class FlowConfig:
    consistency_frames_threshold: int = 3
    mask_magnitude_threshold: float = 0.6
    mask_min_region_area: int = 80


@dataclass(slots=True)
class FusionConfig:
    iou_threshold: float = 0.3


@dataclass(slots=True)
class RuntimeConfig:
    no_motion_reset_frames: int = 5
    alarm_cooldown_frames: int = 0
    sync_tolerance_ms: int = 100


@dataclass(slots=True)
class TriageConfig:
    frame_diff: FrameDiffConfig = field(default_factory=FrameDiffConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
