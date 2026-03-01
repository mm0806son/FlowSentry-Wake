# FlowSentry 场景验证 / Scenario Validation

[![中文](https://img.shields.io/badge/语言-中文-red)](#cn) [![English](https://img.shields.io/badge/Language-English-blue)](#en)

更新时间 / Updated: 2026-03-01

---

<a id="cn"></a>
## 中文

本文档介绍如何用三种常见场景快速验证 FlowSentry 是否工作正常。

### 1. 适用场景

- 固定摄像头
- RTSP 视频源可用
- 已完成基础安装（见 `docs/tutorials/install.md`）

### 2. 验证前准备

```bash
source venv/bin/activate
export RTSP="rtsp://<user>:<pass>@<ip>:554/av_stream/ch0"
```

可选：先确认脚本可用

```bash
python scripts/flowsentry_scenario_runner.py --help
```

### 3. 三种推荐验证场景

脚本：`scripts/flowsentry_scenario_runner.py`

#### 3.1 normal（正常入侵）

用途：有人进入画面并移动，系统应触发报警。

```bash
python scripts/flowsentry_scenario_runner.py \
  edgeflownet-opticalflow-raw \
  yolov8s-coco \
  "$RTSP" \
  --scenario-type normal \
  --duration 15 \
  --notes "single person walk-through"
```

期望结果：`alarm_triggered`

#### 3.2 false_positive（非人目标运动）

用途：画面有运动但不是人，系统应避免误报。

```bash
python scripts/flowsentry_scenario_runner.py \
  edgeflownet-opticalflow-raw \
  yolov8s-coco \
  "$RTSP" \
  --scenario-type false_positive \
  --duration 20 \
  --notes "moving objects without person"
```

期望结果：`no_alarm`

#### 3.3 adversarial（挑战条件）

用途：有人遮挡自身进入场景，系统应触发报警。

```bash
python scripts/flowsentry_scenario_runner.py \
  edgeflownet-opticalflow-raw \
  yolov8s-coco \
  "$RTSP" \
  --scenario-type adversarial \
  --duration 30 \
  --notes "occlusion + slow movement"
```

期望结果：`alarm_triggered`

### 4. 如何查看结果

每次运行会在 `artifacts/flowsentry/scenarios/` 下生成场景目录，主要文件：

- `manifest.json`：本次场景验证结果
- `video.mp4`：对应录像（用于回看）

重点查看 `manifest.json`：

- `expected_result`：该场景期望结果
- `actual_result`：实际结果
- `alarm_triggered`：是否触发报警

### 5. 判定通过标准（用户版）

1. `normal`：触发报警
2. `false_positive`：不触发报警
3. `adversarial`：触发报警

若三类场景都符合，说明系统在“基本可用”状态。

### 6. 常见问题

1. 运行报 source 错误：
- 检查 `RTSP` 地址是否可访问。

2. 没有输出文件：
- 确认脚本运行完成且有写入 `artifacts/flowsentry/scenarios/` 权限。

3. 结果不稳定：
- 保持摄像头固定，先在光照稳定环境下重复验证。

### 7. 进一步调参与开发

本页不包含开发/测试细节。

如需参数调优、脚本测试或故障定位，请参考：

- `dev.md`
- `docs/flowsentry_homeassistant_homepod_integration.md`

---

<a id="en"></a>
## English

This document explains how to quickly validate FlowSentry with three common scenarios.

### 1. Applicable Conditions

- Fixed camera
- RTSP source is available
- Base installation is complete (see `docs/tutorials/install.md`)

### 2. Preparation

```bash
source venv/bin/activate
export RTSP="rtsp://<user>:<pass>@<ip>:554/av_stream/ch0"
```

Optional: verify script availability

```bash
python scripts/flowsentry_scenario_runner.py --help
```

### 3. Three Recommended Validation Scenarios

Script: `scripts/flowsentry_scenario_runner.py`

#### 3.1 normal

Purpose: A person enters the scene and moves. The system should trigger an alarm.

```bash
python scripts/flowsentry_scenario_runner.py \
  edgeflownet-opticalflow-raw \
  yolov8s-coco \
  "$RTSP" \
  --scenario-type normal \
  --duration 15 \
  --notes "single person walk-through"
```

Expected result: `alarm_triggered`

#### 3.2 false_positive

Purpose: Motion exists but not from a person. The system should avoid false alarms.

```bash
python scripts/flowsentry_scenario_runner.py \
  edgeflownet-opticalflow-raw \
  yolov8s-coco \
  "$RTSP" \
  --scenario-type false_positive \
  --duration 20 \
  --notes "moving objects without person"
```

Expected result: `no_alarm`

#### 3.3 adversarial

Purpose: A person enters while self-occluded. The system should still trigger an alarm.

```bash
python scripts/flowsentry_scenario_runner.py \
  edgeflownet-opticalflow-raw \
  yolov8s-coco \
  "$RTSP" \
  --scenario-type adversarial \
  --duration 30 \
  --notes "occlusion + slow movement"
```

Expected result: `alarm_triggered`

### 4. How to Read Results

Each run creates a scenario directory under `artifacts/flowsentry/scenarios/` with:

- `manifest.json`: validation result of this run
- `video.mp4`: recorded evidence

Check these key fields in `manifest.json`:

- `expected_result`
- `actual_result`
- `alarm_triggered`

### 5. Pass Criteria

1. `normal`: alarm is triggered
2. `false_positive`: no alarm
3. `adversarial`: alarm is triggered

If all three match, the system is considered basically operational.

### 6. FAQ

1. Source error at runtime:
- Verify RTSP URL accessibility.

2. No output files:
- Ensure the script finished and has write permission to `artifacts/flowsentry/scenarios/`.

3. Unstable results:
- Keep camera fixed and validate under stable lighting first.

### 7. Further Tuning and Development

This page excludes development/testing details.

For tuning, script tests, and troubleshooting, see:

- `dev.md`
- `docs/flowsentry_homeassistant_homepod_integration.md`
