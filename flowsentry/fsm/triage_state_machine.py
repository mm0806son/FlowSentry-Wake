from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TriageState(str, Enum):
    STANDBY = "standby"
    FLOW_ACTIVE = "flow_active"
    YOLO_VERIFY = "yolo_verify"
    ALARM = "alarm"


@dataclass(frozen=True)
class TriageStepResult:
    state: TriageState
    transitioned: bool
    optical_flow_enabled: bool
    yolo_enabled: bool
    no_motion_frames: int


class TriageStateMachine:
    def __init__(self, no_motion_reset_frames: int = 5) -> None:
        if no_motion_reset_frames < 1:
            raise ValueError("no_motion_reset_frames must be >= 1")
        self.no_motion_reset_frames = no_motion_reset_frames
        self.state = TriageState.STANDBY
        self._no_motion_frames = 0

    @property
    def no_motion_frames(self) -> int:
        return self._no_motion_frames

    def reset(self) -> None:
        self.state = TriageState.STANDBY
        self._no_motion_frames = 0

    def _track_motion_absence(self, flow_present: bool) -> None:
        if flow_present:
            self._no_motion_frames = 0
        else:
            self._no_motion_frames += 1

    def _reset_to_standby_if_idle(self) -> None:
        if self._no_motion_frames >= self.no_motion_reset_frames:
            self.state = TriageState.STANDBY
            self._no_motion_frames = 0

    def step(
        self,
        *,
        frame_diff_triggered: bool,
        flow_present: bool,
        flow_threshold_reached: bool,
        alarm_triggered: bool,
    ) -> TriageStepResult:
        prev_state = self.state

        if self.state == TriageState.STANDBY:
            self._no_motion_frames = 0
            if frame_diff_triggered:
                self.state = TriageState.FLOW_ACTIVE

        elif self.state == TriageState.FLOW_ACTIVE:
            self._track_motion_absence(flow_present)
            if flow_threshold_reached:
                self.state = TriageState.ALARM if alarm_triggered else TriageState.YOLO_VERIFY
            else:
                self._reset_to_standby_if_idle()

        elif self.state == TriageState.YOLO_VERIFY:
            self._track_motion_absence(flow_present)
            if alarm_triggered:
                self.state = TriageState.ALARM
            else:
                self._reset_to_standby_if_idle()

        elif self.state == TriageState.ALARM:
            self._track_motion_absence(flow_present)
            self._reset_to_standby_if_idle()

        return TriageStepResult(
            state=self.state,
            transitioned=self.state != prev_state,
            optical_flow_enabled=self.state != TriageState.STANDBY,
            yolo_enabled=self.state in {TriageState.YOLO_VERIFY, TriageState.ALARM},
            no_motion_frames=self._no_motion_frames,
        )
