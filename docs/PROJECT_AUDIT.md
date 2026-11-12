# 精简版检查：2026-11-12

这是本地交付版检查，尚未推送到GitHub。上次线上核验为commit `b68e943b409a24eece9652d391f5185364b49783`，262文件；`provenance/github_audit.json`只记录该旧版，不代表线上已有本次更改。

## 已完成的精简与验证

- 删除`research_snapshot`全部56文件、ROBIN四文件、无关ControlNet备用/深度/退化/其他采样器代码、MAT训练/指标入口、冗余历史诊断证据。具体删除列表见 [simplification.json](../provenance/simplification.json)。
- 当前生成封装仅保留普通ControlNet步骤。真实GPU重新生成随包clean/bit0/bit1，**三张PNG逐字节一致**，解码系数和消息一致，见 [GPU验证](../provenance/minimal_gpu_validation.json)。
- CPU历史三状态输入/接收/指标汇总回归通过；新稳定模块的顺序、缺失视图、漂移及持久化回归通过。打包最终检查见 [validation.json](../provenance/validation.json)。
- 可选MAT为独立普通领域修复入口；严格加载领域checkpoint及原协议，验证范围见 [MAT验证](../provenance/minimal_mat_validation.json)。无消息约束优化或RGB收益声明。
- 权重、数据下载地址保留在 [DOWNLOADS](DOWNLOADS.md)；原MAT Release不变，不重新拆分/上传。

## 新“先筛区域、再提曲线”开发检查

复用6个既有Cal内容的36张历史clean探测图，先筛原边缘像素，随后才提有序曲线、复核整段并检查实际ST2输入闭环：

| COCO内容ID | 原边缘像素 | 稳定像素 | 提取曲线 | 整段复核通过 | ST2输入闭环通过 |
|---|---:|---:|---:|---:|---:|
| 4595 | 12486 | 10814 | 77 | 47 | 45 |
| 102306 | 34839 | 28706 | 110 | 54 | 34 |
| 149157 | 14127 | 7016 | 34 | 25 | 21 |
| 275339 | 23966 | 16017 | 88 | 36 | 29 |
| 379022 | 29331 | 19711 | 104 | 48 | 30 |
| 564879 | 48813 | 13202 | 30 | 9 | 4 |

六个内容都找到可写候选；这只是开发可运行性，非全新内容可编码率估计，也非消息恢复率。原36张RGB位于原工作区，不额外打包；随包记录见 [region_first_validation.json](../provenance/region_first_validation.json)。`select`也已在实际筛选结果上执行并通过。

## 为什么仍不能称历史整批完整复现

旧代码依赖历史outputs中的selection、曲线samples、码本、ROI、Cal模型、生成计划/噪声及父冻结清单。上一版只放报告/汇总及少量输入，甚至旧脚本`--help`也会因为导入时读取缺失协议而失败。下载ControlNet、COCO和MAT推理权重不能补齐这些实验状态。

现在可运行的是独立主线；75%只能从历史判决重算，不能冒充整批重新生成。新筛选顺序还更换了选择分布，且轻量S/4A接收器不等于旧稳定实验的4B SourceRef。下一步需冻结新协议、补独立内容与留出prompt/seed、重新校准并生成RGB评测。空环境安装、完整历史批次重跑、多载体长消息/RM/RS实图纠错、仅Mask可靠恢复均未完成。

更新已有GitHub仓库必须同步删除旧文件，见 [UPDATE_GITHUB](UPDATE_GITHUB.md)。只覆盖上传不会实现线上精简。
