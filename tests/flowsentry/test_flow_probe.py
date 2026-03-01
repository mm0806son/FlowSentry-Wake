# tests/flowsentry/test_flow_probe.py
# Copyright 2025, FlowSentry-Wake

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from flowsentry.config import FlowConfig
from flowsentry.runtime.adapters import MockFlowBackend
from flowsentry.runtime.flow_probe import (
    FlowProbeLatencySummary,
    FlowProbePreflight,
    FlowProbeRecord,
    FlowProbeRunner,
    FlowProbeSummary,
    summarize_frame_age_samples,
    summarize_flow_probe_records,
    write_flow_probe_jsonl,
    write_flow_probe_summary_json,
)


class TestFlowProbeRecord:
    def test_record_creation(self):
        """Create flow probe record"""
        rec = FlowProbeRecord(
            frame_index=1,
            stream_id=0,
            flow_present=True,
            flow_consistent=True,
            flow_bbox=(10.0, 20.0, 100.0, 200.0),
            flow_region_count=1,
        )
        assert rec.frame_index == 1
        assert rec.flow_present is True
        assert rec.flow_bbox == (10.0, 20.0, 100.0, 200.0)

    def test_record_to_dict(self):
        """Record to dict serialization"""
        rec = FlowProbeRecord(
            frame_index=5,
            stream_id=None,
            flow_present=False,
            flow_consistent=False,
            flow_bbox=None,
            flow_region_count=0,
        )
        d = rec.to_dict()
        assert d["frame_index"] == 5
        assert d["flow_present"] is False
        assert d["flow_bbox"] is None

    def test_record_frozen(self):
        """Record is immutable"""
        rec = FlowProbeRecord(
            frame_index=1,
            stream_id=0,
            flow_present=True,
            flow_consistent=True,
            flow_bbox=None,
            flow_region_count=0,
        )
        with pytest.raises(Exception):
            rec.flow_present = False


class TestFlowProbeSummary:
    def test_summary_creation(self):
        """Create flow probe summary"""
        summary = FlowProbeSummary(
            frames_processed=10,
            frames_with_flow=5,
            frames_with_consistent_flow=3,
            total_flow_regions=15,
            max_flow_regions=5,
            first_flow_frame_index=2,
            stream_ids=(0,),
        )
        assert summary.frames_processed == 10
        assert summary.frames_with_flow == 5

    def test_summary_to_dict(self):
        """Summary to dict serialization"""
        summary = FlowProbeSummary(
            frames_processed=10,
            frames_with_flow=5,
            frames_with_consistent_flow=3,
            total_flow_regions=15,
            max_flow_regions=5,
            first_flow_frame_index=2,
            stream_ids=(0, 1),
        )
        d = summary.to_dict()
        assert d["frames_processed"] == 10
        assert d["stream_ids"] == [0, 1]


class TestFlowProbeLatencySummary:
    def test_summarize_empty_samples(self):
        summary = summarize_frame_age_samples([])
        assert summary == FlowProbeLatencySummary(
            sampled_frames=0,
            p50_ms=None,
            p95_ms=None,
            max_ms=None,
        )

    def test_summarize_samples(self):
        summary = summarize_frame_age_samples([0.2, 0.4, 0.6, 0.8, 1.0])
        assert summary.sampled_frames == 5
        assert summary.p50_ms == 600.0
        assert summary.p95_ms == 960.0
        assert summary.max_ms == 1000.0


class TestFlowProbeRunner:
    def test_runner_with_mock_backend(self):
        """Runner with mock backend"""
        backend = MockFlowBackend(batches=[
            (True, True, (10, 10, 50, 50)),
            (False, False, None),
            (True, False, (20, 20, 60, 60)),
        ])

        @dataclass
        class FakeFrameResult:
            stream_id: int = 0

        stream = [FakeFrameResult(), FakeFrameResult(), FakeFrameResult()]
        runner = FlowProbeRunner(flow_backend=backend, max_frames=3)
        records = list(runner.run(stream))

        assert len(records) == 3
        assert records[0].flow_present is True
        assert records[0].flow_consistent is True
        assert records[1].flow_present is False
        assert records[2].flow_present is True
        assert records[2].flow_consistent is False

    def test_runner_max_frames(self):
        """Runner respects max_frames"""
        backend = MockFlowBackend(batches=[
            (True, True, (10, 10, 50, 50)),
        ] * 10)

        @dataclass
        class FakeFrameResult:
            stream_id: int = 0

        stream = [FakeFrameResult()] * 100
        runner = FlowProbeRunner(flow_backend=backend, max_frames=5)
        records = list(runner.run(stream))

        assert len(records) == 5

    def test_runner_invalid_max_frames(self):
        """Runner rejects invalid max_frames"""
        with pytest.raises(ValueError):
            FlowProbeRunner(flow_backend=MockFlowBackend(), max_frames=0)


class TestSummarizeFlowProbeRecords:
    def test_summarize_empty(self):
        """Summarize empty list"""
        summary = summarize_flow_probe_records([])
        assert summary.frames_processed == 0
        assert summary.frames_with_flow == 0
        assert summary.first_flow_frame_index is None

    def test_summarize_with_records(self):
        """Summarize with records"""
        records = [
            FlowProbeRecord(1, 0, True, True, (10, 10, 50, 50), 2),
            FlowProbeRecord(2, 0, True, False, (20, 20, 60, 60), 1),
            FlowProbeRecord(3, 0, False, False, None, 0),
            FlowProbeRecord(4, 0, True, True, (30, 30, 70, 70), 3),
        ]
        summary = summarize_flow_probe_records(records)
        assert summary.frames_processed == 4
        assert summary.frames_with_flow == 3
        assert summary.frames_with_consistent_flow == 2
        assert summary.total_flow_regions == 6
        assert summary.max_flow_regions == 3
        assert summary.first_flow_frame_index == 1


class TestWriteFlowProbeJsonl:
    def test_write_jsonl(self, tmp_path):
        """Write JSONL file"""
        records = [
            FlowProbeRecord(1, 0, True, True, (10, 10, 50, 50), 1),
            FlowProbeRecord(2, 0, False, False, None, 0),
        ]
        out_path = write_flow_probe_jsonl(records, tmp_path / "flow.jsonl")
        assert out_path.exists()
        content = out_path.read_text()
        assert '"frame_index": 1' in content
        assert '"flow_present": false' in content


class TestWriteFlowProbeSummaryJson:
    def test_write_summary_json(self, tmp_path):
        """Write summary JSON file"""
        summary = FlowProbeSummary(
            frames_processed=10,
            frames_with_flow=5,
            frames_with_consistent_flow=3,
            total_flow_regions=15,
            max_flow_regions=5,
            first_flow_frame_index=2,
            stream_ids=(0,),
        )
        out_path = write_flow_probe_summary_json(summary, tmp_path / "summary.json")
        assert out_path.exists()
        content = out_path.read_text()
        assert '"frames_processed": 10' in content


class TestFlowProbePreflight:
    def test_preflight_creation(self):
        """Create preflight info"""
        preflight = FlowProbePreflight(
            requested_network="edgeflownet-opticalflow",
            resolved_network="/path/to/edgeflownet-opticalflow.yaml",
            source="rtsp://localhost/stream",
            pipe_type="gst",
        )
        assert preflight.requested_network == "edgeflownet-opticalflow"
        assert preflight.pipe_type == "gst"

    def test_preflight_to_dict(self):
        """Preflight to dict serialization"""
        preflight = FlowProbePreflight(
            requested_network="edgeflownet",
            resolved_network="/path/to/edgeflownet.yaml",
            source=None,
            pipe_type="gst",
        )
        d = preflight.to_dict()
        assert d["requested_network"] == "edgeflownet"
        assert d["source"] is None
