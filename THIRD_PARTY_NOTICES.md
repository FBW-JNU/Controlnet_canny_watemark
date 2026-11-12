# 来源、许可证与本次整理修改

本文件夹是多来源研究代码的整理包，保留各源码中的作者与许可证声明。根目录`LICENSE`为原ControlNet仓库的Apache License 2.0文本，不替代下述独立许可。

| 来源 | 本包路径 | 来源/许可说明 |
|---|---|---|
| lllyasviel/ControlNet及工作区修改 | `vendor/controlnet/cldm`、配置 | https://github.com/lllyasviel/ControlNet ，Apache-2.0，附原LICENSE |
| LDM/Stable Diffusion代码 | `vendor/controlnet/ldm` | 当前ControlNet工作区携带版本；保留文件头中的CompVis/OpenAI等来源，不把checkpoint许可等同源码许可 |
| 本工作区ControlNetROBINPipeline | `vendor/controlnet/controlnet_robin` | 当前实验生成封装；保留FFT/adapter作为导入依赖，几何主线不启用。与官方ROBIN完整算法不可混同 |
| MAT | `vendor/MAT` | https://github.com/fenglinglwb/MAT ，CC BY-NC 4.0，附完整LICENSE；相关torch_utils/dnnlib保留原文件头许可 |
| 研究脚本及轻量提取 | `research_snapshot`、`canny_geometry` | 来源文件名/哈希在provenance中；提取的数值函数保持原体，新增IO包装显式区分 |
| 示例 | `examples` | 既有实验的Canny条件、生成RGB与评估面板；不含原始COCO源照片，也不作为独立验证集 |
| COCO筛选元数据 | `data/coco` | COCO 2017 captions和本地Canny条件统计的派生清单；不含源照片。标注遵循COCO的CC BY 4.0说明，原照片版权依各自来源；见https://cocodataset.org/#termsofuse |

本次整理日期2026-11-12，导出副本的变更：

- 将数值函数提取到独立模块，避免旧实验目录的导入副作用；函数体不改。
- ControlNet导出副本`ldm/modules/encoders/modules.py`将两个机器专属CLIP路径改为环境变量`CANNY_WM_CLIP_PATH`，默认回落至构造器version；未改原工作区。
- 新增CLI/IO、生成入口、消息指标汇总、文档和发布完整性检查。
- 精简证据JSON替换工作区绝对路径；报告中未随包携带的原产物链接转换为文字说明。

权重单独获取并遵守其发布许可，下载入口见`docs/DOWNLOADS.md`。MAT领域适配权重尚无可确认的公开下载链接，本次另附独立本地资产ZIP，保留MAT的CC BY-NC 4.0许可和来源记录；代码ZIP仅包含其架构、调用代码和哈希。完整论文PDF、数据集源照片、第三方模型checkpoint未放入代码ZIP。
