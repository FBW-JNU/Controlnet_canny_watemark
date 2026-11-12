# 当前代码地图

| 文件 | 用途 |
|---|---|
| `canny_geometry/__main__.py` | `probe-generate/probe/select/embed/generate/decode/demo/metrics`入口 |
| `canny_geometry/generation.py` | 普通冻结ControlNet；多探测加载一次模型；配对共享CPU噪声 |
| `canny_geometry/controlnet_pipeline.py` | 仅模型加载、conditioning、schedule、DDIM普通步骤；已去除ROBIN依赖 |
| `canny_geometry/stability.py` | 像素支持/漂移统计→稳定边缘图→有序曲线→整段复核 |
| `canny_geometry/matcher.py` | 细化、图追踪、分段、样条、去重、路径匹配 |
| `canny_geometry/api.py` | S/ST编码、实际输入准入、参考码本、读码及消息指标 |
| `canny_geometry/geometry.py`、`raster.py` | 规范方向/法向与实际二值像素移动 |
| `canny_geometry/observations.py`、`receiver.py` | 输出局部对应、联合拟合、S/4A判决与软信息 |
| `canny_geometry/mat_repair.py` | 独立可选普通领域MAT修复，支持native/refined及PiDiNet转换；无消息优化 |
| `canny_geometry/channel.py`、`benchmark.py` | 复用普通ControlNet，冻结整批输入、重生成、独立解码与输入核对 |
| `canny_geometry/historical_receiver.py`、`historical_calibration.py` | 精确历史4B接收器，4A Cal模型重建 |
| `canny_geometry/research.py`、`quality.py` | 新流程prepare/select/calibrate/report及配对质量四指标 |
| `tools/compare_reproduction.py`、`diagnose_reproduction.py` | 与公布金标准比对；失败消息与配对clean的路径/门槛诊断 |
| `tools/run_reproduction.py` | 历史整批或新流程完整运行、独立重复比较 |
| `benchmarks/` | 相对路径冻结输入、计划、参考及新协议 |
| `configs/cldm_v15.yaml` | 当前SD1.5 Canny模型结构 |
| `vendor/controlnet/cldm/`、`ldm/` | 当前生成配置所需网络、VAE、CLIP与DDIM依赖 |
| `vendor/MAT/` | MAT推理网络和StyleGAN算子依赖；保留原许可证/内核源码 |
| `tools/smoke_check.py` | 历史单例编码/解码和逐图判决汇总回归 |
| `tools/check_stability.py` | 先筛后提顺序、缺失探测、漂移与统计输出回归 |
| `tools/restore_coco_manifest.py` | 将随包normal总体清单映射到新机器 |
| `tools/seal_release.py`、`verify_bundle.py` | 发布检查、逐文件哈希和代码ZIP |

数值函数来源见 [source_manifest.json](../provenance/source_manifest.json)；移除文件列表见 [simplification.json](../provenance/simplification.json)。原研究工作区未改动。此包不再交付依赖历史outputs的原批量脚本；冻结评测已移植为独立benchmark入口，范围见复现文档。

普通MAT保留为后续分支的推理基础，不代表旧W+消息约束实验已迁移。主线接收器与旧稳定实验的4B SourceRef也须区分，见 [复现范围](REPRODUCIBILITY.md)。
