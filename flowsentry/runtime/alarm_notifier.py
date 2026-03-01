from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any, Callable
from urllib import error, request
import json


FLOW_NO_OBJECT_REASON = "flow_no_object_detected"


@dataclass(frozen=True)
class AlarmEvent:
    alarm_flag: bool
    alarm_reason: str
    frame_index: int
    state: str
    timestamp_iso: str
    stream_id: int | None = None
    camera: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "alarm_flag": bool(self.alarm_flag),
            "alarm_reason": self.alarm_reason,
            "frame_index": int(self.frame_index),
            "state": self.state,
            "timestamp": self.timestamp_iso,
        }
        if self.stream_id is not None:
            payload["stream_id"] = int(self.stream_id)
        if self.camera:
            payload["camera"] = self.camera
        return payload


@dataclass(frozen=True)
class HaWebhookConfig:
    enabled: bool = False
    url: str | None = None
    timeout_seconds: float = 2.0
    cooldown_seconds: float = 10.0
    no_object_delay_frames: int = 5
    allowed_reasons: tuple[str, ...] = (
        "person_iou_match",
        "person_iou_below_threshold",
        FLOW_NO_OBJECT_REASON,
    )


@dataclass(frozen=True)
class NotifyResult:
    sent: bool
    reason: str
    status_code: int | None = None
    error: str | None = None


class HaAlarmNotifier:
    def __init__(
        self,
        config: HaWebhookConfig,
        *,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        self.config = config
        self._now_fn = now_fn or monotonic
        self._last_sent_at_by_reason: dict[str, float] = {}
        self._flow_no_object_streak = 0

    def _post_json(self, payload: dict[str, Any]) -> int:
        if not self.config.url:
            raise ValueError("Webhook URL is not configured")
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.config.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
            return int(getattr(resp, "status", 200))

    def _update_flow_no_object_streak(self, event: AlarmEvent) -> None:
        if event.alarm_flag and event.alarm_reason == FLOW_NO_OBJECT_REASON:
            self._flow_no_object_streak += 1
            return
        self._flow_no_object_streak = 0

    def notify_if_needed(self, event: AlarmEvent) -> NotifyResult:
        self._update_flow_no_object_streak(event)

        if not self.config.enabled:
            return NotifyResult(sent=False, reason="disabled")
        if not self.config.url:
            return NotifyResult(sent=False, reason="missing_webhook_url")
        if not event.alarm_flag:
            return NotifyResult(sent=False, reason="alarm_flag_false")
        if event.alarm_reason not in self.config.allowed_reasons:
            return NotifyResult(sent=False, reason="reason_not_allowed")

        if event.alarm_reason == FLOW_NO_OBJECT_REASON:
            required = max(1, int(self.config.no_object_delay_frames))
            if self._flow_no_object_streak < required:
                return NotifyResult(sent=False, reason="flow_no_object_delay")

        now = float(self._now_fn())
        cooldown = max(0.0, float(self.config.cooldown_seconds))
        last_sent = self._last_sent_at_by_reason.get(event.alarm_reason)
        if cooldown > 0 and last_sent is not None and (now - last_sent) < cooldown:
            return NotifyResult(sent=False, reason="cooldown")

        try:
            status = self._post_json(event.to_payload())
        except (ValueError, error.URLError, TimeoutError, OSError) as exc:
            return NotifyResult(sent=False, reason="send_failed", error=str(exc))

        self._last_sent_at_by_reason[event.alarm_reason] = now
        return NotifyResult(sent=True, reason="sent", status_code=status)
