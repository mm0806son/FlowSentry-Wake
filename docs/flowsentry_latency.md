# Flow Probe 实时延迟优化手册 / Flow Probe Real-Time Latency Optimization Guide

[![中文](https://img.shields.io/badge/语言-中文-red)](#cn) [![English](https://img.shields.io/badge/Language-English-blue)](#en)

更新时间 / Updated: 2026-03-01

---

<a id="cn"></a>
## 中文

### 1. 文档目的

本手册用于指导用户在 Flow Probe 场景下完成实时延迟优化，建立可复现的低延迟运行基线。

### 2. 适用范围

- 硬件平台：Orange Pi + Metis + 固定摄像头
- 视频输入：RTSP
- 脚本入口：`scripts/run_flowsentry_flow_probe.py`

### 3. 推荐基线

建议统一使用以下参数作为初始配置：

- `frame-rate=7`
- `rtsp-latency=100`
- DMA 开启（默认）

执行命令：

```bash
source venv/bin/activate
unset AXELERA_USE_DMABUF || true

python scripts/run_flowsentry_flow_probe.py \
  edgeflownet-opticalflow-raw \
  "$RTSP" \
  --pipe gst \
  --frames 800 \
  --summary-only \
  --rtsp-latency 100
```

### 4. 参数说明

#### 4.1 frame-rate

- 作用：限制输入处理帧率，避免推理吞吐与输入帧率不匹配导致积压。
- 推荐值：`7`
- 说明：`0` 表示跟随源帧率，仅用于对照，不建议作为生产默认。

#### 4.2 rtsp-latency

- 作用：设置 RTSP 管线缓冲。
- 推荐值：`100`
- 备选值：`200`
- 说明：`500` 往往带来更高整体延迟，`50` 容易导致尾延迟波动。

#### 4.3 DMA

- 作用：降低数据搬运开销。
- 建议：保持默认开启。
- 排障策略：仅在兼容性问题出现时使用 `--no-use-dmabuf` 做对照。

### 5. 标准调优流程

#### 步骤 1：建立基线

```bash
python scripts/run_flowsentry_flow_probe.py \
  edgeflownet-opticalflow-raw \
  "$RTSP" \
  --pipe gst \
  --frames 300 \
  --summary-only \
  --rtsp-latency 100 \
  --frame-rate 7
```

#### 步骤 2：单变量调优 rtsp-latency

在 `frame-rate=7` 保持不变时，分别测试：

- `--rtsp-latency 100`
- `--rtsp-latency 200`

选择 `p95` 更稳定的一组。

#### 步骤 3：验证 frame-rate 影响

```bash
# A: 推荐配置
python scripts/run_flowsentry_flow_probe.py edgeflownet-opticalflow-raw "$RTSP" \
  --pipe gst --frames 300 --summary-only --rtsp-latency 100 --frame-rate 7

# B: 对照配置
python scripts/run_flowsentry_flow_probe.py edgeflownet-opticalflow-raw "$RTSP" \
  --pipe gst --frames 300 --summary-only --rtsp-latency 100 --frame-rate 0
```

若 B 的尾延迟高于 A，应固定使用 `frame-rate=7`。

### 6. 达标判据

建议以稳定性为主，不以单次峰值作为结论。

达标要求：

1. 连续多次运行中 `p95` 保持稳定。
2. 画面无持续性拖尾。
3. 不出现长时间积压。

建议测试帧数：

- 快速评估：`300` 帧
- 现场确认：`800` 帧

### 7. 常见问题处理

#### 7.1 延迟突然升高

1. 检查网络抖动与摄像头码率。
2. 检查 `frame-rate` 是否被改为 `0`。
3. 恢复推荐基线复测。

#### 7.2 调参后效果不明显

1. 每次仅修改一个参数。
2. 每组参数至少运行 `300` 帧。
3. 使用同一时段和同一视频源对比。

#### 7.3 兼容性异常

1. 保持推荐参数不变，先确认基础链路可运行。
2. 必要时关闭 DMA 做对照：`--no-use-dmabuf`。
3. 对照结果记录后再决定是否保留变更。

### 8. 结论

在当前平台与典型 RTSP 场景下，`frame-rate=7 + rtsp-latency=100 + DMA 默认开启` 是可复现且稳定的低延迟基线配置。

---

<a id="en"></a>
## English

### 1. Purpose

This guide helps users optimize real-time latency in Flow Probe and establish a reproducible low-latency baseline.

### 2. Scope

- Hardware: Orange Pi + Metis + fixed camera
- Video source: RTSP
- Script entry: `scripts/run_flowsentry_flow_probe.py`

### 3. Recommended Baseline

Use the following configuration as the initial baseline:

- `frame-rate=7`
- `rtsp-latency=100`
- DMA enabled (default)

Command:

```bash
source venv/bin/activate
unset AXELERA_USE_DMABUF || true

python scripts/run_flowsentry_flow_probe.py \
  edgeflownet-opticalflow-raw \
  "$RTSP" \
  --pipe gst \
  --frames 800 \
  --summary-only \
  --rtsp-latency 100
```

### 4. Parameter Notes

#### 4.1 frame-rate

- Purpose: limits input processing FPS to prevent queue buildup when source FPS exceeds inference throughput.
- Recommended value: `7`
- Note: `0` follows source FPS and is for comparison only.

#### 4.2 rtsp-latency

- Purpose: sets RTSP pipeline buffering.
- Recommended value: `100`
- Alternative: `200`
- Note: `500` typically increases latency, while `50` may worsen tail jitter.

#### 4.3 DMA

- Purpose: reduces data transfer overhead.
- Recommendation: keep it enabled by default.
- Troubleshooting: use `--no-use-dmabuf` only for compatibility diagnostics.

### 5. Standard Tuning Procedure

#### Step 1: Build baseline

```bash
python scripts/run_flowsentry_flow_probe.py \
  edgeflownet-opticalflow-raw \
  "$RTSP" \
  --pipe gst \
  --frames 300 \
  --summary-only \
  --rtsp-latency 100 \
  --frame-rate 7
```

#### Step 2: Tune rtsp-latency with single-variable changes

Keep `frame-rate=7` fixed and test:

- `--rtsp-latency 100`
- `--rtsp-latency 200`

Choose the one with more stable `p95`.

#### Step 3: Validate frame-rate impact

```bash
# A: recommended configuration
python scripts/run_flowsentry_flow_probe.py edgeflownet-opticalflow-raw "$RTSP" \
  --pipe gst --frames 300 --summary-only --rtsp-latency 100 --frame-rate 7

# B: comparison configuration
python scripts/run_flowsentry_flow_probe.py edgeflownet-opticalflow-raw "$RTSP" \
  --pipe gst --frames 300 --summary-only --rtsp-latency 100 --frame-rate 0
```

If B shows higher tail latency than A, keep `frame-rate=7`.

### 6. Acceptance Criteria

Prioritize stability over single peak values.

Acceptance requirements:

1. `p95` remains stable across repeated runs.
2. No sustained visual lag.
3. No long-term frame backlog.

Recommended frame counts:

- Quick evaluation: `300` frames
- Site confirmation: `800` frames

### 7. Troubleshooting

#### 7.1 Sudden latency increase

1. Check network jitter and camera bitrate changes.
2. Verify `frame-rate` was not changed to `0`.
3. Re-run with baseline settings.

#### 7.2 No visible improvement after tuning

1. Change only one parameter at a time.
2. Run at least `300` frames per parameter set.
3. Compare under the same source and time window.

#### 7.3 Compatibility issues

1. Keep baseline settings and verify core pipeline first.
2. Disable DMA only for comparison: `--no-use-dmabuf`.
3. Decide on final settings after recording comparison results.

### 8. Conclusion

For the current platform and typical RTSP conditions, `frame-rate=7 + rtsp-latency=100 + DMA enabled` is a reproducible and stable low-latency baseline.
