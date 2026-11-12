# 代码清单与建议阅读顺序

`research_snapshot/` 保存原实验与本地依赖的逐字节快照。完整文件名列表见 [snapshot_scripts.json](../provenance/snapshot_scripts.json)，每个文件来源/哈希见 [source_manifest.json](../provenance/source_manifest.json)。原脚本根目录在本工作区，文件名不变。

## 实际使用的 ControlNet

| 文件 | 作用 |
|---|---|
| `vendor/controlnet/cldm/cldm.py` | ControlNet、ControlledUnetModel、ControlLDM；含当前工作区已有修改 |
| `configs/cldm_v15.yaml` | 当前 SD1.5 Canny 网络配置 |
| `vendor/controlnet/cldm/model.py` | 配置实例化、checkpoint 加载 |
| `vendor/controlnet/cldm/ddim_hacked.py` | DDIM schedule；原 register_buffer 偏向默认 CUDA，入口按实验同样替换成模型设备 |
| `vendor/controlnet/cldm/hack.py` | 原实验使用的 CLIP 文本编码补丁 |
| `vendor/controlnet/ldm/` | LatentDiffusion、VAE、UNet、attention、文本编码等实际依赖 |
| `vendor/controlnet/controlnet_robin/pipeline.py` | 当前生成封装；S/S-T只调用 get_conditioning、make_schedule、ddim_step、VAE decode |
| `vendor/controlnet/controlnet_robin/adapter.py`、`fft_watermark.py` | pipeline导入依赖，当前几何实验不启用其水印/adapter功能 |
| `canny_geometry/generation.py` | 可移植生成入口，保持当前20步/CFG5/共享噪声配置 |

当前 checkpoint 不是 SD2.1/SDXL，也不是ControlNet训练任务。权重名称与哈希见复现文档。

## 曲线主线

以下路径均相对 `research_snapshot/`。

| 阶段 | 文件 | 作用 |
|---|---|---|
| A/B | `canny_curve_stage1_32.py` | 32内容探测计划与生成 |
| A/B | `analyze_canny_curve_stage1_32.py`、`frozen_stage1_matcher.py` | 细化/图路径候选、曲线匹配、发现集选择 |
| A/B | `report_canny_curve_stage1_32.py`、`audit_canny_curve_stage1_32.py` | 报告与发现/留出分离检查 |
| S编码 | `curve_broad_peak_code.py` | S/T基函数、联合Huber拟合、R0/A3候选与码本分数 |
| 栅格编辑 | `curve_qim_closed_loop_v2.py` 中 `RasterWarp` | 实际二值像素移动、取整、邻接重连；S分支只复用栅格器 |
| 几何参考 | `canny_curve_stage2_bit.py`、`curve_receiver_stage2c.py` | 方向、弧长、法向、输入局部观测 |
| 对应 | `curve_receiver_stage2b.py` | RGB Canny、路径采样、允许跳站的有序观测 |
| 1bit | `experiment_s_frozen_validation.py` | Fresh24输入可写选择、参考码本、三状态配对生成 |
| 1bit评价 | `analyze_s_frozen_validation.py`、`report_s_frozen_validation.py`、`audit_s_frozen_validation.py` | 冻结分类、擦除/质量诊断与审计 |
| 2bit | `curve_single_2bit_stage4a.py` | S/T四符号、九状态输入检查、Cal信道、Eval判决 |
| 2bit评价 | `report_single_2bit_stage4a.py`、`audit_single_2bit_stage4a.py` | 完整消息与逐位统计 |
| 稳定选择 | `canny_st_stable_carrier_compare.py` | 原方法 vs 六张clean探测筛选的配对对照 |
| 稳定报告 | `report_st_stable_carrier_compare.py` | 逐内容、错误/擦除、质量及配对差异 |
| 幅度对照 | `canny_st_amplitude_cal_scan.py`、`analyze_st_amplitude_cal_scan.py`、`report_st_amplitude_cal_scan.py` | 1.5/2/3/4px，同噪声与内容，LOCO评价 |

## 部分缺失、参考消融与后续纠错

| 文件 | 作用 |
|---|---|
| `curve_partial_st_stage4b.py` | ObsDrop已匹配观测删点、EdgeDrop删边重匹配；联合信息、条件数、不确定度 |
| `calibrate_partial_st_stage4b.py` | Cal门槛选择，包含修订记录依赖 |
| `diagnose_partial_st_stage4b.py`、`report_partial_st_stage4b.py`、`audit_partial_st_stage4b.py` | 缺失区域/位置诊断、质量与审计 |
| `compare_st_identifiability_4b.py`、`report_st_identifiability_4b.py` | 同PNG接收器对比、消offset信息、码字距离、多候选歧义 |
| `apply_verified_oracle_st_4b.py` | 有真实核验对应标注时的重放接口；包中没有伪造对应标注 |
| `curve_reference_ablation_stage4c0.py`、`report_reference_ablation_stage4c0.py` | 去clean参考、公共码本、管道/矩形Mask和全图自参考消融 |
| `s_frozen_offline_codes.py` | 重复8次与RM(1,3)离线符号模拟，无实图多载体 |
| `s_key_consistency_offline.py` | 历史离线一致性模拟，当前消息恢复主任务非必需 |
| `trace_qim_input_signal.py`、`repair_qim_input_closure.py` | 历史QIM输入取整/零修改诊断，非当前S/S-T主编码 |

## MAT 分支

| 文件 | 作用 |
|---|---|
| `mat_st_message_experiment.py` | G0原图、G1普通MAT、G2事后投影、G3消息约束W+再投影、G4直接编辑 |
| `mat_st_message_constraints_v2.py` | 阈值对齐BCE、面积与系数损失、细线投影修订 |
| `mat_st_message_soft_compare.py` | 二值MAT与旧PiDiNet-soft native/refined权重配对比较 |
| `evaluate_mat_st_message_experiment.py`、`report_mat_st_message_soft_compare.py` | 实际PNG闭环、输入门槛后才允许扩散、报告 |
| `audit_mat_st_soft_projection_support.py` | 检查投影支撑是否实际来自缺口内部 |
| `fuxian_papers/scripts/mat_common.py` | MAT加载与参考PyTorch算子 |
| `fuxian_papers/scripts/mat_canny_model.py` | RGB权重向灰度架构转换与修复封装 |
| `fuxian_papers/scripts/mat_hed_pidi_repair.py` | native/refined RepairMAT与转换器架构；当前使用已有领域权重 |
| `vendor/MAT/`（包根目录） | 原MAT上游源码，独立许可证 |

原MAT脚本引用`fuxian_papers/MAT`及旧领域训练产物。包中的上游源码位于`vendor/MAT`，未改写冻结脚本路径；完整重跑需按复现文档恢复资产布局。

## 轻量入口与来源

`canny_geometry/matcher.py`、`geometry.py`、`raster.py`、`observations.py`、`receiver.py`从上表精确提取数值函数，只重新声明本地导入，避免导入时读取历史outputs。提取函数名和原始行号在文件内，来源哈希在manifest中。

`api.py` 和 `__main__.py` 是本次整理新增的IO/判决包装；`probe`只筛稳定候选，后续仍须做`embed`输入准入。它们不是已经跑过新独立验证的统一研究系统；4B/4C/MAT细节以原脚本为准。

## 权重与数据入口

- `docs/DOWNLOADS.md`：公开预训练权重和COCO的下载地址、命令、目录与复现边界。
- `provenance/external_assets.json`：模型文件大小、已知哈希、公开URL或独立资产包说明。
- `data/coco/normal_manifest.csv.gz`：原94,625行normal总体，保留行序和prompt。
- `tools/restore_coco_manifest.py`：使用标准库恢复新机器上的绝对图像路径，可检查全部图像存在，拒绝覆盖。
- 独立`canny-geometry-mat-domain-weights_20261112.zip`：4个当前MAT消息推理权重及原始协议/阈值，未放入代码ZIP。
