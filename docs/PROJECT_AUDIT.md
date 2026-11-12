# 本地交付版检查：2026-11-12

本次新增冻结批次复现，不改原工作区脚本、历史输出或权重。当前GitHub仓库不会自动获得这些新增文件；本次代码ZIP与线上版本的差异见交付说明。

## 精简范围

历史脚本快照、ROBIN四文件、FFT/adapter、无关ControlNet分支和MAT训练入口继续移除。普通ControlNet所需上游网络及可选MAT推理依赖保留。原删除清单见[simplification.json](../provenance/simplification.json)。新增`benchmarks`是重建结果必需的输入与协议，未包含完整COCO、模型权重或整批RGB。

## 整批复现检查

- 历史S1、4A及旧稳定选择评测已迁移到相对路径清单和独立接收器。生成仅使用条件、prompt/seed及固定模型；预期哈希和消息标签只用于评价。
- 原历史RGB另做独立解码、质量和4A Cal重建，见[历史资产审计](../provenance/historical_original_audit.json)。实际重生成的数量、比对结果及新流程评价见[复现验收](REPRODUCTION_RESULT.md)与[机器可读证据](../provenance/full_reproduction.json)。
- 新流程固定6 Cal+12 Eval：先原E的clean探测，像素筛选后才提曲线；全部选择先冻结，Cal校准先于Eval生成。无载体计擦除且保留分母。独立重复从同一原始E重新运行。
- CPU回归覆盖真实4A/4B RGB接收、空观测擦除、输入篡改拒绝、通道配置校验、区域筛选顺序与ROI；最终打包检查见[validation.json](../provenance/validation.json)。

## 复现与研究效果的边界

冻结评测的重建、完整流程的重复和消息恢复准确率分别验收，不能互相替代。历史75%只属于旧“先提曲线再筛”的六Cal内容和4B参考接收器。新流程使用独立内容与重新校准的S/4A接收器，不能把新旧数字差异直接归因于筛选顺序。

可选MAT仍是普通修复推理；消息约束优化、多载体长消息、RM/RS实图纠错、仅Mask可靠恢复尚未交付。没有验证空环境在线安装、所有硬件数值一致或MAT从零重训。

权重和数据地址见[DOWNLOADS](DOWNLOADS.md)。原MAT Release独立权重ZIP保持不变。运行命令见[REPRODUCIBILITY](REPRODUCIBILITY.md)。
