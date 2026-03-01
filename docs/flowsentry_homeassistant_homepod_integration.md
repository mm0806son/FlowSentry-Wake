# FlowSentry Alarm Integration with Home Assistant + HomePod

[![中文](https://img.shields.io/badge/语言-中文-red)](#cn) [![English](https://img.shields.io/badge/Language-English-blue)](#en)

更新时间 / Updated: 2026-03-01

---

<a id="cn"></a>
## 中文

### 1. 这份文档解决什么问题

当 FlowSentry 触发报警时，把报警事件通过 Home Assistant Webhook 转成 HomePod 的语音/音频播报。

链路：
`FlowSentry alarm event -> Home Assistant webhook -> media_player.play_media -> HomePod`

### 2. 使用前准备

你需要先准备：

1. FlowSentry 运行环境已可正常拉取 RTSP 视频流。
2. Home Assistant 可访问，且已接入 HomePod（存在 `media_player` 实体）。
3. 告警音频文件可被 HA 访问，例如：`config/www/audio/alarm.mp3`。
4. 可访问的音频 URL，例如：`http://<ha-ip>:8123/local/audio/alarm.mp3`。

### 3. Home Assistant 端配置

在 Home Assistant 新建一个 `webhook` 触发的自动化（把以下示例中的占位符替换为你的值）：

```yaml
alias: FlowSentry Alarm -> HomePod
triggers:
  - trigger: webhook
    webhook_id: <your-webhook-id>
    allowed_methods:
      - POST
    local_only: true
actions:
  - action: media_player.volume_set
    target:
      entity_id: media_player.your_homepod_entity
    data:
      volume_level: 0.7
  - action: media_player.play_media
    target:
      entity_id: media_player.your_homepod_entity
    data:
      media_content_type: music
      media_content_id: "http://<ha-ip>:8123/local/audio/alarm.mp3"
mode: single
```

Webhook URL 格式：

```text
http://<ha-ip>:8123/api/webhook/<your-webhook-id>
```

### 4. 在 FlowSentry 中启用 HA 推送

```bash
source venv/bin/activate
export RTSP="rtsp://<user>:<pass>@<ip>:554/av_stream/ch0"
export HA="http://<ha-ip>:8123/api/webhook/<your-webhook-id>"

python scripts/run_flowsentry_dual_probe.py \
  edgeflownet-opticalflow-raw \
  yolov8s-coco \
  "$RTSP" \
  --pipe gst \
  --frames 100 \
  --display \
  --overlay \
  --rtsp-latency 100 \
  --frame-rate 7 \
  --magnitude-threshold 6 \
  --min-region-area 120 \
  --ha-webhook-enabled \
  --ha-webhook-url "$HA" \
  --ha-webhook-timeout 2.0 \
  --ha-webhook-cooldown-seconds 10 \
  --ha-no-object-delay-frames 5 \
  --ha-camera-name front_gate
```

说明：

- `--ha-webhook-enabled`：开启 HA 推送。
- `--ha-webhook-url`：目标 webhook 地址（也可用环境变量 `HA` / `HA_WEBHOOK_URL`）。
- `--ha-webhook-cooldown-seconds`：同类报警最小推送间隔，避免频繁播报。
- `--ha-no-object-delay-frames`：`flow_no_object_detected` 连续多少帧才推送，默认 `5`。
- `--ha-camera-name`：写入 payload 的摄像头名称，便于 HA 自动化区分来源。

### 5. 会触发播报的报警类型

默认建议只在以下原因推送给 HA：

- `flow_no_object_detected`
- `person_iou_match`

通常不推送：

- `non_person_object_detected`
- `person_iou_below_threshold`
- `no_flow`

### 6. Payload 说明（示例）

FlowSentry 推送给 HA 的 payload 示例：

```json
{
  "alarm_flag": true,
  "camera": "front_gate",
  "level": "alarm",
  "frame_index": 123,
  "alarm_reason": "person_iou_match",
  "state": "ALARM",
  "timestamp": "2026-03-01T10:00:00+00:00"
}
```

其中最常用于自动化判断的字段是：`alarm_flag`、`alarm_reason`、`camera`。

### 7. 常见问题

1. HomePod 没有声音：
- 先确认 `media_player` 实体 ID 正确。
- 检查 `media_content_id` URL 在 HA 所在网络可访问。

2. FlowSentry 报警了但 HA 无动作：
- 检查 `--ha-webhook-enabled` 是否开启。
- 检查 webhook URL（`HA`）是否与自动化 `webhook_id` 一致。

3. 告警太频繁：
- 提高 `--ha-webhook-cooldown-seconds`。
- 保持 `--ha-no-object-delay-frames` 为 `5` 或更大。

---

<a id="en"></a>
## English

### 1. What This Guide Solves

When FlowSentry triggers an alarm, it sends an event to a Home Assistant webhook, which then triggers HomePod audio playback.

Flow:
`FlowSentry alarm event -> Home Assistant webhook -> media_player.play_media -> HomePod`

### 2. Prerequisites

Before starting, make sure:

1. FlowSentry can read your RTSP stream.
2. Home Assistant is reachable and your HomePod is available as a `media_player` entity.
3. Alarm audio is available in HA, for example: `config/www/audio/alarm.mp3`.
4. The audio URL is reachable, for example: `http://<ha-ip>:8123/local/audio/alarm.mp3`.

### 3. Home Assistant Configuration

Create a webhook-triggered automation in Home Assistant (replace placeholders with your values):

```yaml
alias: FlowSentry Alarm -> HomePod
triggers:
  - trigger: webhook
    webhook_id: <your-webhook-id>
    allowed_methods:
      - POST
    local_only: true
actions:
  - action: media_player.volume_set
    target:
      entity_id: media_player.your_homepod_entity
    data:
      volume_level: 0.7
  - action: media_player.play_media
    target:
      entity_id: media_player.your_homepod_entity
    data:
      media_content_type: music
      media_content_id: "http://<ha-ip>:8123/local/audio/alarm.mp3"
mode: single
```

Webhook URL format:

```text
http://<ha-ip>:8123/api/webhook/<your-webhook-id>
```

### 4. Enable HA Push in FlowSentry

```bash
source venv/bin/activate
export RTSP="rtsp://<user>:<pass>@<ip>:554/av_stream/ch0"
export HA="http://<ha-ip>:8123/api/webhook/<your-webhook-id>"

python scripts/run_flowsentry_dual_probe.py \
  edgeflownet-opticalflow-raw \
  yolov8s-coco \
  "$RTSP" \
  --pipe gst \
  --frames 100 \
  --display \
  --overlay \
  --rtsp-latency 100 \
  --frame-rate 7 \
  --magnitude-threshold 6 \
  --min-region-area 120 \
  --ha-webhook-enabled \
  --ha-webhook-url "$HA" \
  --ha-webhook-timeout 2.0 \
  --ha-webhook-cooldown-seconds 10 \
  --ha-no-object-delay-frames 5 \
  --ha-camera-name front_gate
```

Parameter notes:

- `--ha-webhook-enabled`: enables HA webhook output.
- `--ha-webhook-url`: target webhook URL (or use `HA` / `HA_WEBHOOK_URL`).
- `--ha-webhook-cooldown-seconds`: minimum interval between repeated alarm pushes.
- `--ha-no-object-delay-frames`: delay frames for `flow_no_object_detected` before sending (default `5`).
- `--ha-camera-name`: camera name included in payload.

### 5. Alarm Reasons Recommended for Playback

Recommended push reasons:

- `flow_no_object_detected`
- `person_iou_match`

Usually not pushed:

- `non_person_object_detected`
- `person_iou_below_threshold`
- `no_flow`

### 6. Payload Example

Example payload sent from FlowSentry to HA:

```json
{
  "alarm_flag": true,
  "camera": "front_gate",
  "level": "alarm",
  "frame_index": 123,
  "alarm_reason": "person_iou_match",
  "state": "ALARM",
  "timestamp": "2026-03-01T10:00:00+00:00"
}
```

Fields most commonly used in HA automation conditions: `alarm_flag`, `alarm_reason`, `camera`.

### 7. FAQ

1. No sound from HomePod:
- Verify `media_player` entity ID.
- Verify the `media_content_id` URL is reachable from HA.

2. FlowSentry alarms but HA does nothing:
- Ensure `--ha-webhook-enabled` is set.
- Ensure webhook URL (`HA`) matches the automation `webhook_id`.

3. Too many repeated alarms:
- Increase `--ha-webhook-cooldown-seconds`.
- Keep `--ha-no-object-delay-frames` at `5` or higher.

## 8. References

- https://www.home-assistant.io/docs/automation/trigger/
- https://www.home-assistant.io/integrations/media_player/
- https://www.home-assistant.io/integrations/apple_tv/
- https://www.music-assistant.io/player-support/airplay/
