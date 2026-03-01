#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from os import getenv
from pathlib import Path
from urllib.parse import urlparse
from urllib import error, request


def build_payload(camera: str, level: str, reason: str, run_id: str, seq: int) -> dict[str, object]:
    return {
        "camera": camera,
        "level": level,
        "alarm_reason": reason,
        "state": "ALARM",
        "event_type": "manual_test",
        "run_id": run_id,
        "sequence": seq,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def post_webhook(
    webhook_url: str,
    payload: dict[str, object],
    timeout: float,
    max_response_bytes: int,
) -> dict[str, object]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        webhook_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with request.urlopen(req, timeout=timeout) as resp:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        raw = resp.read(max_response_bytes).decode("utf-8", errors="replace")
        return {
            "status": resp.status,
            "reason": getattr(resp, "reason", ""),
            "headers": dict(resp.headers.items()),
            "body": raw,
            "elapsed_ms": round(elapsed_ms, 2),
        }


def append_jsonl(path: str | None, record: dict[str, object]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_api_url_from_webhook(webhook_url: str, api_path: str) -> str:
    parsed = urlparse(webhook_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid webhook URL: {webhook_url}")
    return f"{parsed.scheme}://{parsed.netloc}{api_path}"


def get_automation_state(base_url: str, token: str, entity_id: str, timeout: float) -> dict[str, object]:
    url = f"{base_url}/api/states/{entity_id}"
    req = request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="GET",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read(65536).decode("utf-8", errors="replace")
        if resp.status != 200:
            raise RuntimeError(f"Unexpected HA status for {entity_id}: {resp.status}")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected HA state payload type: {type(data)}")
        return data


def resolve_webhook_url_default() -> str:
    return (
        getenv("HA")
        or getenv("HA_WEBHOOK_URL")
        or "http://<ha-ip>:8123/api/webhook/<webhook-id>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send minimal FlowSentry alarm test event to Home Assistant webhook",
    )
    parser.add_argument(
        "--webhook-url",
        default=resolve_webhook_url_default(),
        help="Home Assistant webhook URL. Default: env HA / HA_WEBHOOK_URL / built-in URL",
    )
    parser.add_argument("--camera", default="front_gate", help="Camera name in payload")
    parser.add_argument("--level", default="alarm", help="Alarm level in payload")
    parser.add_argument("--reason", default="manual_webhook_test", help="Alarm reason in payload")
    parser.add_argument("--timeout", type=float, default=2.0, help="HTTP timeout in seconds")
    parser.add_argument("--count", type=int, default=1, help="How many events to send (default: 1)")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between events")
    parser.add_argument(
        "--max-response-bytes",
        type=int,
        default=4096,
        help="Max response bytes to read from HA",
    )
    parser.add_argument(
        "--log-jsonl",
        help="Optional JSONL path to save request/response logs",
    )
    parser.add_argument(
        "--ha-token",
        default=getenv("HA_TOKEN"),
        help="Home Assistant long-lived token. Defaults to env HA_TOKEN.",
    )
    parser.add_argument(
        "--verify-automation-entity-id",
        help="Optional automation entity_id for trigger verification, e.g. automation.flowsentry_alarm_homepod",
    )
    parser.add_argument(
        "--verify-wait-seconds",
        type=float,
        default=1.0,
        help="Wait time before reading automation state after webhook send",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print payload only, do not send HTTP request",
    )
    args = parser.parse_args()

    if args.count < 1:
        raise SystemExit("--count must be >= 1")
    if args.interval < 0:
        raise SystemExit("--interval must be >= 0")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be > 0")
    if args.max_response_bytes < 1:
        raise SystemExit("--max-response-bytes must be >= 1")
    if args.verify_wait_seconds < 0:
        raise SystemExit("--verify-wait-seconds must be >= 0")
    if args.verify_automation_entity_id and not args.ha_token:
        raise SystemExit(
            "--verify-automation-entity-id requires --ha-token or env HA_TOKEN",
        )

    print(f"Webhook URL: {args.webhook_url}")
    run_id = f"ha-test-{int(time.time())}"
    failures = 0
    verify_before: dict[str, object] | None = None
    verify_after: dict[str, object] | None = None
    base_url = build_api_url_from_webhook(args.webhook_url, "")

    if args.verify_automation_entity_id and not args.dry_run:
        try:
            verify_before = get_automation_state(
                base_url=base_url,
                token=args.ha_token,
                entity_id=args.verify_automation_entity_id,
                timeout=args.timeout,
            )
            print(
                "[VERIFY-BEFORE] "
                f"entity={args.verify_automation_entity_id} "
                f"state={verify_before.get('state')!r} "
                f"last_triggered={verify_before.get('attributes', {}).get('last_triggered')!r}"
            )
        except Exception as e:
            failures += 1
            print(f"[VERIFY-ERROR] before-send failed: {e}")

    for seq in range(1, args.count + 1):
        payload = build_payload(
            camera=args.camera,
            level=args.level,
            reason=args.reason,
            run_id=run_id,
            seq=seq,
        )
        event_log: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seq": seq,
            "run_id": run_id,
            "request": {
                "url": args.webhook_url,
                "payload": payload,
                "timeout_s": args.timeout,
            },
        }
        if args.dry_run:
            print(f"[DRY-RUN] seq={seq} payload={json.dumps(payload, ensure_ascii=False)}")
            event_log["result"] = "dry_run"
            append_jsonl(args.log_jsonl, event_log)
        else:
            try:
                response = post_webhook(
                    args.webhook_url,
                    payload,
                    args.timeout,
                    args.max_response_bytes,
                )
                event_log["result"] = "ok"
                event_log["response"] = response
                print(
                    f"[OK] seq={seq} status={response['status']} "
                    f"reason={response['reason']!r} elapsed_ms={response['elapsed_ms']}"
                )
                print(f"  headers={json.dumps(response['headers'], ensure_ascii=False)}")
                print(f"  body={response['body']!r}")
                append_jsonl(args.log_jsonl, event_log)
            except error.HTTPError as e:
                failures += 1
                body = e.read(args.max_response_bytes).decode("utf-8", errors="replace")
                event_log["result"] = "http_error"
                event_log["response"] = {
                    "status": e.code,
                    "reason": getattr(e, "reason", ""),
                    "headers": dict(e.headers.items()) if e.headers else {},
                    "body": body,
                }
                print(
                    f"[HTTP-ERROR] seq={seq} status={e.code} "
                    f"reason={getattr(e, 'reason', '')!r} body={body!r}"
                )
                append_jsonl(args.log_jsonl, event_log)
            except error.URLError as e:
                failures += 1
                event_log["result"] = "url_error"
                event_log["error"] = {"reason": repr(e.reason)}
                print(f"[URL-ERROR] seq={seq} reason={e.reason!r}")
                append_jsonl(args.log_jsonl, event_log)
            except TimeoutError:
                failures += 1
                event_log["result"] = "timeout"
                print(f"[TIMEOUT] seq={seq} timeout={args.timeout}s")
                append_jsonl(args.log_jsonl, event_log)

        if seq < args.count and args.interval > 0:
            time.sleep(args.interval)

    if args.verify_automation_entity_id and not args.dry_run:
        if args.verify_wait_seconds > 0:
            time.sleep(args.verify_wait_seconds)
        try:
            verify_after = get_automation_state(
                base_url=base_url,
                token=args.ha_token,
                entity_id=args.verify_automation_entity_id,
                timeout=args.timeout,
            )
            print(
                "[VERIFY-AFTER] "
                f"entity={args.verify_automation_entity_id} "
                f"state={verify_after.get('state')!r} "
                f"last_triggered={verify_after.get('attributes', {}).get('last_triggered')!r}"
            )
            before_last = (
                verify_before.get("attributes", {}).get("last_triggered")
                if isinstance(verify_before, dict)
                else None
            )
            after_last = (
                verify_after.get("attributes", {}).get("last_triggered")
                if isinstance(verify_after, dict)
                else None
            )
            if after_last and after_last != before_last:
                print("[VERIFY] automation appears triggered by this test.")
            else:
                print(
                    "[VERIFY] automation trigger not observed from state diff. "
                    "Check webhook_id, mode, and HA traces."
                )
        except Exception as e:
            failures += 1
            print(f"[VERIFY-ERROR] after-send failed: {e}")

    total = args.count
    success = total - failures
    print(f"Summary: total={total}, success={success}, failed={failures}, run_id={run_id}")
    print(
        "Note: webhook HTTP 200 from HA does NOT prove trigger success; "
        "invalid webhook_id can also return 200."
    )
    if failures > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
