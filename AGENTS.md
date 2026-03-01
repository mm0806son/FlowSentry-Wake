# OpenCode Agent 初始化总结

## 基本信息与角色定义

我是 OpenCode，一个专门为 FlowSentry-Wake 项目服务的 AI 开发助手。基于 .cursor 目录下的规则和发现文档，我已经完成了初始化，以下是关键配置和项目状态总结。

## 项目上下文

### FlowSentry-Wake 项目定位
- **项目名称**: FlowSentry-Wake
- **项目目标**: 面向低功耗安防场景的自适应 AI 哨兵系统
- **硬件平台**: Orange Pi 5 Plus + Metis M.2 加速器 + 固定摄像头
- **核心价值**: 低功耗、低误报、抗对抗/非标准目标鲁棒性

### 核心架构（四阶段分层感知）
1. **Stage 1 Standby**: 低功耗运动检测（MOG2/帧差法）
2. **Stage 2 Triage**: 光流/YOLO 快速筛查
3. **Stage 3 Fusion**: 双流融合与报警决策
4. **Stage 4 Overlay/场景验证**: 可视化与验证

## 项目完成状态

### 已完成功能
| Stage | 状态 | 完成内容 |
|-------|------|----------|
| Stage 1 | ✅ Complete | Triage 骨架、状态机（STANDBY→FLOW_ACTIVE→YOLO_VERIFY→ALARM）、IoU 匹配、报警策略 |
| Stage 2 | ✅ Complete | YOLO 真实接入（AxeleraYoloBackend）、probe 脚本、RTSP 验证通过（180 帧）、验收脚本 |
| Stage 3 | ✅ Complete | EdgeFlowNet 接入（AxeleraFlowBackend）、光流 probe、双流融合（process_with_both_backends） |
| Stage 4 | ✅ Complete | Overlay 可视化（OverlayRenderer）、双流 probe、场景验证脚本、视频保存 |

### 代码统计
- **新增模块**: `flowsentry/`（motion、fusion、fsm、runtime、overlay、vision）
- **新增脚本**: 6 个 CLI 工具（yolo_probe、flow_probe、dual_probe、stage2_acceptance、scenario_runner、triage_replay）
- **测试覆盖**: 210 passed, 1 skipped
- **文档**: 实施方案、场景验证指南、进度记录

## 开发规范与工作流

### 语言规则
- 默认使用中文回复
- 面向用户的 Markdown 文档默认使用中文
- 代码、标识符与代码注释默认使用英文

### 实施流程
1. **Understand**: 先读现有实现与相邻功能
2. **Test**: 优先先写测试（红灯）
3. **Implement**: 用最小改动让测试通过（绿灯）
4. **Refactor**: 在测试通过前提下清理结构
5. **Commit**: 提交信息明确说明 "why"，并关联计划文档

### 复杂任务处理
- 多步骤任务开始前先检查 `.cursor/task_plan.md`、`.cursor/progress.md`、`.cursor/findings.md`
- 涉及 3 个及以上步骤时，先写 `.cursor/Implementation_{feature_name}.md`（中文）
- 统一在 `.cursor/IMPLEMENTATION_PLAN.md` 维护分阶段计划，不在仓库根目录创建 `IMPLEMENTATION_PLAN.md`

### 质量门禁
- 每次提交前必须满足：可编译、测试通过、格式化与 lint 通过
- 新功能或修复必须带测试
- 不保留无 issue 编号的 TODO

## 环境配置

### 虚拟环境
- 当前使用项目根目录 `venv/` 虚拟环境
- 激活命令: `source venv/bin/activate`
- 验证命令: `venv/bin/python -m pytest tests/flowsentry -q`

### 关键工具与模块
- **光流模型部署与推理**: EdgeFlowNet 相关模块
- **YOLO 检测**: Axelera YOLO 后端适配器
- **框匹配基础能力**: `axelera/app/model_utils/box.py`
- **多 pipeline 编排**: `examples/multiple_pipelines.py`

## 当前状态与下一步

### 已完成核心功能
- 运动检测: 帧差触发 + 光流连续计数
- 人员检测: YOLO person bbox 提取
- 融合报警: IoU 匹配 + 连续帧阈值 + 报警策略
- 可视化: 实时 overlay + 视频保存
- 验证工具: probe 脚本 + 场景运行器

### 待现场验证
1. Stage 3: 真实 RTSP 上光流 bbox 稳定性
2. Stage 4: 三类场景录制（normal/false_positive/adversarial）

### 后续建议
1. **参数调优**: 根据实际场景调整 `consistency_frames_threshold`、`iou_threshold`、`person_conf`
2. **性能优化**: 帧率、延迟、内存占用监控
3. **功能扩展**: 多摄像头管理、报警通知、监控统计（根据实际需求）

## 项目归档状态

- 核心功能已实现，项目结束
- 后续迭代可在当前分支基础上继续开发
- 会话恢复时读取 `.cursor/` 目录下的文档

---

*初始化完成时间: 2026-02-27*
*项目状态: FlowSentry-Wake 核心 Stage 1-4 完成*
