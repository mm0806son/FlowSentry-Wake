from __future__ import annotations

from flowsentry.runtime.alarm_notifier import (
    AlarmEvent,
    FLOW_NO_OBJECT_REASON,
    HaAlarmNotifier,
    HaWebhookConfig,
)


def _event(*, flag: bool = True, reason: str = "person_iou_match", frame: int = 1) -> AlarmEvent:
    return AlarmEvent(
        alarm_flag=flag,
        alarm_reason=reason,
        frame_index=frame,
        state="ALARM",
        timestamp_iso="2026-02-28T21:00:00+08:00",
        stream_id=0,
        camera="front_gate",
    )


def test_notifier_sends_when_alarm_flag_true_and_reason_allowed(monkeypatch):
    notifier = HaAlarmNotifier(HaWebhookConfig(enabled=True, url="http://ha/webhook", cooldown_seconds=0.0))
    monkeypatch.setattr(notifier, "_post_json", lambda _payload: 200)

    result = notifier.notify_if_needed(_event())

    assert result.sent is True
    assert result.reason == "sent"
    assert result.status_code == 200


def test_notifier_skips_when_alarm_flag_false(monkeypatch):
    notifier = HaAlarmNotifier(HaWebhookConfig(enabled=True, url="http://ha/webhook", cooldown_seconds=0.0))
    monkeypatch.setattr(notifier, "_post_json", lambda _payload: 200)

    result = notifier.notify_if_needed(_event(flag=False))

    assert result.sent is False
    assert result.reason == "alarm_flag_false"


def test_notifier_skips_when_reason_not_allowed(monkeypatch):
    notifier = HaAlarmNotifier(HaWebhookConfig(enabled=True, url="http://ha/webhook", cooldown_seconds=0.0))
    monkeypatch.setattr(notifier, "_post_json", lambda _payload: 200)

    result = notifier.notify_if_needed(_event(reason="non_person_object_detected"))

    assert result.sent is False
    assert result.reason == "reason_not_allowed"


def test_flow_no_object_reason_uses_default_five_frame_delay(monkeypatch):
    notifier = HaAlarmNotifier(HaWebhookConfig(enabled=True, url="http://ha/webhook", cooldown_seconds=0.0))
    sent = []
    monkeypatch.setattr(notifier, "_post_json", lambda _payload: sent.append(1) or 200)

    for idx in range(1, 5):
        result = notifier.notify_if_needed(_event(reason=FLOW_NO_OBJECT_REASON, frame=idx))
        assert result.sent is False
        assert result.reason == "flow_no_object_delay"

    result5 = notifier.notify_if_needed(_event(reason=FLOW_NO_OBJECT_REASON, frame=5))
    assert result5.sent is True
    assert result5.reason == "sent"
    assert len(sent) == 1


def test_notifier_applies_cooldown_by_reason(monkeypatch):
    now = {"t": 100.0}
    notifier = HaAlarmNotifier(
        HaWebhookConfig(enabled=True, url="http://ha/webhook", cooldown_seconds=10.0, no_object_delay_frames=1),
        now_fn=lambda: now["t"],
    )
    monkeypatch.setattr(notifier, "_post_json", lambda _payload: 200)

    first = notifier.notify_if_needed(_event(reason=FLOW_NO_OBJECT_REASON, frame=1))
    second = notifier.notify_if_needed(_event(reason=FLOW_NO_OBJECT_REASON, frame=2))
    now["t"] = 111.0
    third = notifier.notify_if_needed(_event(reason=FLOW_NO_OBJECT_REASON, frame=3))

    assert first.sent is True
    assert second.sent is False
    assert second.reason == "cooldown"
    assert third.sent is True


def test_notifier_send_failure_is_non_blocking(monkeypatch):
    notifier = HaAlarmNotifier(HaWebhookConfig(enabled=True, url="http://ha/webhook", cooldown_seconds=0.0))

    def _raise(_payload):
        raise OSError("network down")

    monkeypatch.setattr(notifier, "_post_json", _raise)
    result = notifier.notify_if_needed(_event())

    assert result.sent is False
    assert result.reason == "send_failed"
    assert "network down" in (result.error or "")
