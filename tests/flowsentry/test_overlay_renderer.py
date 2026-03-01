from __future__ import annotations

import numpy as np
import pytest

from flowsentry.fusion.iou_matcher import IoUMatchResult
from flowsentry.overlay.config import OverlayConfig
from flowsentry.overlay.renderer import OverlayRenderer
from flowsentry.runtime.orchestrator import TriageOrchestratorOutput
from flowsentry.fsm.triage_state_machine import TriageState
from flowsentry.types import AlarmDecision, FrameSignals


@pytest.fixture
def sample_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_signals():
    return FrameSignals(
        frame_diff_triggered=True,
        flow_present=True,
        flow_consistent=True,
        flow_bbox=(100, 100, 200, 200),
        person_bboxes=((150, 150, 250, 250),),
        timestamp_ms=1234567890,
    )


@pytest.fixture
def sample_result():
    return TriageOrchestratorOutput(
        state=TriageState.ALARM,
        optical_flow_enabled=True,
        yolo_enabled=True,
        consistency_count=3,
        flow_threshold_reached=True,
        match=IoUMatchResult(
            matched=True,
            best_iou=0.75,
            best_person_bbox=(150, 150, 250, 250),
            threshold=0.5,
        ),
        alarm=AlarmDecision(
            triggered=True,
            reason="person_iou_match",
            best_iou=0.75,
            matched=True,
        ),
    )


def test_overlay_renderer_init():
    renderer = OverlayRenderer()
    assert renderer.config is not None


def test_overlay_renderer_with_custom_config():
    config = OverlayConfig(
        state_color=(255, 0, 0),
        flow_bbox_color=(0, 255, 0),
        person_bbox_color=(0, 0, 255),
    )
    renderer = OverlayRenderer(config=config)
    assert renderer.config.state_color == (255, 0, 0)
    assert renderer.config.flow_bbox_color == (0, 255, 0)
    assert renderer.config.person_bbox_color == (0, 0, 255)


def test_overlay_renderer_returns_frame(sample_frame, sample_signals, sample_result):
    renderer = OverlayRenderer()
    annotated = renderer.render(sample_frame, sample_signals, sample_result)
    
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == sample_frame.shape
    assert annotated.dtype == np.uint8


def test_overlay_renderer_does_not_modify_original(
    sample_frame, sample_signals, sample_result
):
    renderer = OverlayRenderer()
    original_frame = sample_frame.copy()
    
    _ = renderer.render(sample_frame, sample_signals, sample_result)
    
    np.testing.assert_array_equal(sample_frame, original_frame)


def test_overlay_renderer_draws_flow_bbox(sample_frame, sample_signals, sample_result):
    renderer = OverlayRenderer()
    annotated = renderer.render(sample_frame, sample_signals, sample_result)
    
    assert not np.array_equal(annotated, sample_frame)


def test_overlay_renderer_draws_person_bbox(sample_frame, sample_result):
    signals = FrameSignals(
        frame_diff_triggered=False,
        flow_present=False,
        flow_consistent=False,
        flow_bbox=None,
        person_bboxes=((150, 150, 250, 250), (300, 300, 400, 400)),
    )
    
    renderer = OverlayRenderer()
    annotated = renderer.render(sample_frame, signals, sample_result)
    
    assert not np.array_equal(annotated, sample_frame)


def test_overlay_renderer_draws_alarm_border(sample_frame, sample_signals, sample_result):
    renderer = OverlayRenderer()
    annotated = renderer.render(sample_frame, sample_signals, sample_result)
    
    assert not np.array_equal(annotated, sample_frame)


def test_overlay_renderer_no_alarm_no_border(sample_frame):
    signals = FrameSignals(
        frame_diff_triggered=False,
        flow_present=False,
        flow_consistent=False,
    )
    result = TriageOrchestratorOutput(
        state=TriageState.STANDBY,
        optical_flow_enabled=False,
        yolo_enabled=False,
        consistency_count=0,
        flow_threshold_reached=False,
        match=IoUMatchResult(
            matched=False,
            best_iou=None,
            best_person_bbox=None,
            threshold=0.5,
        ),
        alarm=AlarmDecision(triggered=False, reason=""),
    )
    
    renderer = OverlayRenderer()
    annotated = renderer.render(sample_frame, signals, result)
    
    assert annotated is not None


def test_overlay_config_defaults():
    config = OverlayConfig()
    
    assert config.state_color is not None
    assert config.flow_bbox_color is not None
    assert config.person_bbox_color is not None
    assert config.alarm_border_color is not None
    assert config.font_scale > 0
    assert config.line_thickness > 0


def test_overlay_renderer_with_none_bboxes(sample_frame, sample_result):
    signals = FrameSignals(
        frame_diff_triggered=False,
        flow_present=False,
        flow_consistent=False,
        flow_bbox=None,
        person_bboxes=(),
    )
    
    renderer = OverlayRenderer()
    annotated = renderer.render(sample_frame, signals, sample_result)
    
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == sample_frame.shape
