# Development Guide (dev.md)

[![中文](https://img.shields.io/badge/语言-中文-red)](#cn-dev) [![English](https://img.shields.io/badge/Language-English-blue)](#en-dev)

This document is for contributors and maintainers.

---

<a id="cn-dev"></a>
## 中文

### 1. 适用对象

本文档面向开发者，包含本地开发、测试、联调与发布前检查。

### 2. 开发环境准备

1. 进入仓库根目录并激活环境

```bash
source venv/bin/activate
```

2. 推荐设置 `PYTHONPATH`

```bash
export PYTHONPATH="$PWD:${PYTHONPATH}"
```

3. 可选：确认关键脚本可用

```bash
python scripts/run_flowsentry_dual_probe.py --help
python scripts/run_flowsentry_flow_probe.py --help
python scripts/run_flowsentry_yolo_probe.py --help
```

### 3. 开发流程（建议）

1. 先阅读相关实现与相邻模块。
2. 先补测试或先写失败用例。
3. 用最小改动实现功能。
4. 回归核心测试。
5. 同步文档与变更记录。

### 4. 测试命令

FlowSentry 全量测试：

```bash
venv/bin/python -m pytest -q tests/flowsentry
```

常用子集：

```bash
venv/bin/python -m pytest -q tests/flowsentry/test_dual_probe.py
venv/bin/python -m pytest -q tests/flowsentry/test_flow_probe_script.py
venv/bin/python -m pytest -q tests/flowsentry/test_yolo_probe_script.py
```

### 5. 联调命令（示例）

双流联调（RTSP）：

```bash
python scripts/run_flowsentry_dual_probe.py \
  edgeflownet-opticalflow-raw yolov8s-coco "$RTSP" \
  --display --overlay
```

场景验证：

```bash
python scripts/flowsentry_scenario_runner.py --help
```

### 6. 第三方 SDK 边界

本仓库包含第三方 Axelera Voyager SDK 目录，开发时请注意边界：

- 第一方自研核心：`flowsentry/`, `scripts/run_flowsentry_*.py`, `tests/flowsentry/`
- 第三方目录：`axelera/`, `ax_models/`, `operators/`, `tools/`, `trackers/`, `licenses/`

非必要情况下，不修改第三方目录中的源码与接口行为。

### 7. 发布前开发检查清单

1. `tests/flowsentry` 回归通过。
2. README 面向用户，开发细节保留在本文件。
3. 敏感信息已脱敏（RTSP、Webhook、Token、内网地址）。
4. 版本与许可证信息已对齐 `RELEASE_NOTES.md`、`LICENSE.txt`、`licenses/`。

---

<a id="en-dev"></a>
## English

### 1. Audience

This document is for developers and maintainers, covering local development, testing, validation, and pre-release checks.

### 2. Environment Setup

1. Enter repo root and activate environment

```bash
source venv/bin/activate
```

2. Recommended `PYTHONPATH`

```bash
export PYTHONPATH="$PWD:${PYTHONPATH}"
```

3. Optional script sanity check

```bash
python scripts/run_flowsentry_dual_probe.py --help
python scripts/run_flowsentry_flow_probe.py --help
python scripts/run_flowsentry_yolo_probe.py --help
```

### 3. Suggested Workflow

1. Read existing implementation and neighboring modules first.
2. Add tests or failing cases before implementation.
3. Implement with minimal changes.
4. Run core regression tests.
5. Sync docs and change records.

### 4. Test Commands

Full FlowSentry test suite:

```bash
venv/bin/python -m pytest -q tests/flowsentry
```

Common subsets:

```bash
venv/bin/python -m pytest -q tests/flowsentry/test_dual_probe.py
venv/bin/python -m pytest -q tests/flowsentry/test_flow_probe_script.py
venv/bin/python -m pytest -q tests/flowsentry/test_yolo_probe_script.py
```

### 5. Validation Commands (Examples)

Dual-stream runtime (RTSP):

```bash
python scripts/run_flowsentry_dual_probe.py \
  edgeflownet-opticalflow-raw yolov8s-coco "$RTSP" \
  --display --overlay
```

Scenario runner:

```bash
python scripts/flowsentry_scenario_runner.py --help
```

### 6. Third-Party SDK Boundary

This repository contains third-party Axelera Voyager SDK directories. Keep boundary clear:

- First-party custom core: `flowsentry/`, `scripts/run_flowsentry_*.py`, `tests/flowsentry/`
- Third-party directories: `axelera/`, `ax_models/`, `operators/`, `tools/`, `trackers/`, `licenses/`

Avoid changing third-party source behavior unless required.

### 7. Pre-Release Developer Checklist

1. `tests/flowsentry` regression passes.
2. User-facing content stays in `README.md`; development details stay here.
3. Sensitive data is removed or masked.
4. Version and license references are aligned with `RELEASE_NOTES.md`, `LICENSE.txt`, and `licenses/`.
