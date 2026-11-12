# 运行与复现边界

## 已随包验证的层级

1. **源码与报告审阅**：无需安装依赖，见代码地图及evidence。
2. **CPU历史示例重放、输入编码、消息汇总**：不需要外部资产；见README快速开始。
3. **GPU新生成**：需要外部ControlNet checkpoint、CLIP目录及兼容环境；本次打包未重跑GPU，不把CPU验证写成完整生成复现。
4. **原批量实验完整复现**：仍需既有探测图、冻结Cal和历史完整输出等依赖。COCO筛选清单已附压缩元数据；MAT领域权重另附独立资产包，均不代表完整历史产物已齐备。

## 环境

几何工具依赖见`requirements-cpu.txt`。当前实际Control环境为Python3.9.23、torch2.0.1+cu118、torchvision0.15.2+cu118、numpy1.24.4、scipy1.13.1、OpenCV4.11.0、Pillow11.3.0。GPU附加依赖见`requirements-generation.txt`。

```bash
conda create -n canny-geometry python=3.9
conda activate canny-geometry
python -m pip install -r requirements-cpu.txt
# 如需GPU生成，再按自己CUDA环境安装；下面对应原实验版本：
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-generation.txt
```

这是原实验环境版本的安装建议，本次未从空环境在线安装。`research_snapshot/environment.original.yaml`是原上游较旧环境参考（torch1.12），不是当前环境锁文件。可用`python tools/check_environment.py --generation`仅检查导入和版本，不下载模型或分配GPU。可选MAT架构额外依赖见`requirements-mat.txt`；权重和数据依然需要单独恢复。

## 外部权重

**可直接点击的下载地址、完整命令、目录和校验值以 [权重与数据下载](DOWNLOADS.md) 为准。** 不必为了主线额外下载 SD1.5、Depth/HED/OpenPose 检测器或 ROBIN 权重。

| 资产 | 获取/路径要求 |
|---|---|
| ControlNet SD1.5 Canny完整checkpoint | 官方ControlNet发布仓库 `https://huggingface.co/lllyasviel/ControlNet` 的 `models/control_sd15_canny.pth`；需遵守权重许可 |
| CLIP文本模型及tokenizer | `openai/clip-vit-large-patch14` 的本地HuggingFace目录；包含config/tokenizer/vocab/权重等 |
| MAT上游Places512 | 当前加载器使用IOPaint发布的`Places_512_FullData_G.pth`，不是原生`.pkl`；见下载文档 |
| MAT Canny/PiDiNet-soft领域权重 | `canny_native.pt`、`pidi_soft_native.pt`、`pidi_soft_refined.pt`、`converter.pt`及原始阈值/协议，见独立MAT资产包；暂无公开URL |
| COCO与内容清单 | 下载train2017及annotations；随包`data/coco/normal_manifest.csv.gz`可映射为原脚本需要的CSV，保留94,625条内容的顺序和prompt |

原ControlNet checkpoint SHA-256：

```text
4de384b16bc2d7a1fb258ca0cbd941d7dd0a721ae996aff89f905299d6923f45
```

轻量生成命令要求显式给本地CLIP路径，避免原代码中的机器专属路径。只修改包内CLIP加载位置；原工作区文件不变。权重或数值环境不同，不能期待PNG逐字节相同。

## 新生成示例

```bash
python -m canny_geometry generate \
  --conditions examples/conditions \
  --prompt "A realistic documentary photograph of the following scene: A young girl talking on a cell phone while wearing a jacket." \
  --seed 2026114101 \
  --checkpoint /YOUR_MODELS/control_sd15_canny.pth \
  --clip /YOUR_MODELS/clip-vit-large-patch14 \
  --device cuda --out runs/generated

python -m canny_geometry decode --reference examples/reference_1bit.json \
  --rgb runs/generated/bit1.png --out runs/generated_bit1.json
```

条件图须512×512，先于编码完成缩放与边缘提取；禁止编码后隐式缩放。生成入口共享同一个CPU初始噪声、模型eval、float32、20步DDIM、CFG5、eta0、control1，无adapter或latent注入；不覆盖已有输出。

## 稳定候选探测

先用原始未修改Canny生成六张探测RGB：三个prompt组，每组两个seed。生成/图像路径由调用者明确提供，不把marked当clean探测。准备`probe_plan.json`：

```json
[
  {"rgb":"probes/p0_s0.png","prompt_group":0},
  {"rgb":"probes/p0_s1.png","prompt_group":0},
  {"rgb":"probes/p1_s0.png","prompt_group":1},
  {"rgb":"probes/p1_s1.png","prompt_group":1},
  {"rgb":"probes/p2_s0.png","prompt_group":2},
  {"rgb":"probes/p2_s1.png","prompt_group":2}
]
```

```bash
python -m canny_geometry probe --canny YOUR_ORIGINAL_CANNY.png \
  --plan probe_plan.json --out runs/stable_candidates.json
```

输出每条候选的samples、稳定合格状态和探测统计。按长度降序取稳定候选，逐条将其samples保存到曲线JSON，再运行`embed`检查实际输入。最终评价用另一组prompt/seed，先冻结载体、码本和接收器，不用评价RGB重新选曲线。

该CLI复用原严格图路径稳定规则，不隐式读取COCO的letterbox ROI，接收的Canny应已经裁好有效区域；若需原批量协议逐像素一致，请运行原脚本并恢复原ROI和全部资产。

## 原脚本的完整性与限制

`research_snapshot/`中的根脚本保持逐字节，不把硬编码路径/冻结哈希静默改成新实验。它们常在导入或freeze时读取`outputs/`协议、清单与模型哈希；因此直接运行`python research_snapshot/xxx.py ...`可能因资产缺失失败，甚至`--help`也有导入依赖。这是为什么包中另有独立轻量入口。

若要完整重跑，需在独立工作区恢复原root结构：将快照根脚本放root，`vendor/controlnet`的包放root，配置放`models/cldm_v15.yaml`；MAT放`fuxian_papers/MAT`，相应脚本放`fuxian_papers/scripts`。恢复数据与旧协议后，使用原报告中的action顺序。不同机器需要显式重建路径和新的冻结记录，不能篡改旧冻结清单后仍称原审计通过。

## 数值与边界检查

`tools/smoke_check.py`对完整图层面比较：CPU历史RGB判决重放、三个实际输入PNG逐像素复现、Mask外不变、稳定配对所有判决的汇总与原summary一致、空RGB返回擦除。没有用自造成功样本替代实验性能。

`tools/verify_bundle.py`只检查打包文件哈希、禁入大资产和完整性，不检验研究成功。随包`provenance/validation.json`记录本次实际执行的验证范围。
