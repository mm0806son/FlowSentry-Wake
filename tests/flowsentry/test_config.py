# tests/flowsentry/test_config.py
# Copyright 2025, FlowSentry-Wake

import pytest
from flowsentry.config import (
    FrameDiffConfig,
    FlowConfig,
    FusionConfig,
    RuntimeConfig,
    TriageConfig,
)


class TestFrameDiffConfig:
    def test_defaults(self):
        cfg = FrameDiffConfig()
        assert cfg.min_area == 400
        assert cfg.pixel_change_ratio_threshold == 0.01

    def test_custom_values(self):
        cfg = FrameDiffConfig(min_area=100, pixel_change_ratio_threshold=0.05)
        assert cfg.min_area == 100
        assert cfg.pixel_change_ratio_threshold == 0.05


class TestFlowConfig:
    def test_defaults(self):
        cfg = FlowConfig()
        assert cfg.consistency_frames_threshold == 3
        assert cfg.mask_magnitude_threshold == 0.6
        assert cfg.mask_min_region_area == 80

    def test_custom_threshold(self):
        cfg = FlowConfig(consistency_frames_threshold=5)
        assert cfg.consistency_frames_threshold == 5


class TestFusionConfig:
    def test_defaults(self):
        cfg = FusionConfig()
        assert cfg.iou_threshold == 0.3

    def test_custom_iou(self):
        cfg = FusionConfig(iou_threshold=0.5)
        assert cfg.iou_threshold == 0.5


class TestRuntimeConfig:
    def test_defaults(self):
        cfg = RuntimeConfig()
        assert cfg.no_motion_reset_frames == 5
        assert cfg.alarm_cooldown_frames == 0
        assert cfg.sync_tolerance_ms == 100


class TestTriageConfig:
    def test_aggregate_config(self):
        cfg = TriageConfig()
        assert isinstance(cfg.frame_diff, FrameDiffConfig)
        assert isinstance(cfg.flow, FlowConfig)
        assert isinstance(cfg.fusion, FusionConfig)
        assert isinstance(cfg.runtime, RuntimeConfig)

    def test_nested_customization(self):
        cfg = TriageConfig()
        cfg.flow.consistency_frames_threshold = 10
        cfg.fusion.iou_threshold = 0.4
        assert cfg.flow.consistency_frames_threshold == 10
        assert cfg.fusion.iou_threshold == 0.4
