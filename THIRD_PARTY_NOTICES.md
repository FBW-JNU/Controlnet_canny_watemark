# 来源、许可证与整理修改

根目录LICENSE保留ControlNet的Apache-2.0文本，不替代各依赖文件头和独立许可。

| 来源 | 包内路径 | 来源/许可 |
|---|---|---|
| ControlNet及工作区修改 | `vendor/controlnet/cldm`、配置 | https://github.com/lllyasviel/ControlNet ，Apache-2.0 |
| LDM/Stable Diffusion | `vendor/controlnet/ldm` | 保留CompVis/OpenAI等原文件头；checkpoint许可另计 |
| MAT及StyleGAN依赖 | `vendor/MAT` | https://github.com/fenglinglwb/MAT ，CC BY-NC4.0及各文件头；附完整LICENSE |
| 几何数值函数与MAT领域架构 | `canny_geometry` | 从原工作区研究代码提取；文件内和provenance保留来源 |
| 生成封装 | `canny_geometry/controlnet_pipeline.py` | 从旧共享封装提取普通步骤，去除全部ROBIN/adapter/FFT调用与导入 |
| 示例 | `examples` | 既有Canny、生成RGB和面板，不是独立验证集 |
| COCO元数据 | `data/coco` | captions/条件统计派生清单，无照片；标注CC BY4.0，照片版权依来源，https://cocodataset.org/#termsofuse |

仅修改交付副本：移除无关旧入口；提取独立推理与几何模块；CLIP机器路径改为`CANNY_WM_CLIP_PATH`；新增区域先筛选、CLI与发布检查。普通ControlNet路径保持原数值步骤。MAT普通修复新封装不代表历史消息约束优化已迁移。

权重下载及其许可见 [DOWNLOADS](docs/DOWNLOADS.md)。MAT领域Release保留原资产、许可证与协议字节；本次未更改。未打包论文PDF、完整数据集、模型checkpoint和虚拟环境。
