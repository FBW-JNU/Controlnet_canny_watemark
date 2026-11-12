# 操作与复现范围

## 环境与历史示例

建议Python3.9。原兼容环境为torch2.0.1+cu118、torchvision0.15.2、numpy1.24.4、scipy1.13.1、OpenCV4.11、Pillow11.3。未验证空环境在线安装。

```bash
python -m pip install -r requirements-cpu.txt
python -m canny_geometry demo --out runs/demo
python tools/smoke_check.py
python tools/check_stability.py
# GPU生成另需：
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-generation.txt
python tools/check_environment.py --generation
```

权重、COCO、MAT下载命令及校验见 [DOWNLOADS](DOWNLOADS.md)。CPU示例不需要COCO和权重。

## 新流程：从原始Canny到读码

先准备512×512二值0/255的原始`E.png`，缩放/裁剪在编码前完成。建立`prompts.json`，三个不同prompt各配两个不同seed，例如：

```json
[
  {"prompt_group":0,"prompt":"A photograph of a bus on a city road.","seed":1001},
  {"prompt_group":0,"prompt":"A photograph of a bus on a city road.","seed":1002},
  {"prompt_group":1,"prompt":"A pencil drawing of a bus on a city road.","seed":1001},
  {"prompt_group":1,"prompt":"A pencil drawing of a bus on a city road.","seed":1002},
  {"prompt_group":2,"prompt":"An oil painting of a bus on a city road.","seed":1001},
  {"prompt_group":2,"prompt":"An oil painting of a bus on a city road.","seed":1002}
]
```

```bash
# 1. 固定原始E完成六次clean探测；模型只加载一次
python -m canny_geometry probe-generate --canny E.png --plan prompts.json \
  --checkpoint models/control_sd15_canny.pth --clip models/clip-vit-large-patch14 \
  --device cuda --out runs/probes
# 2. 先筛原始边缘像素，然后才提曲线
python -m canny_geometry probe --canny E.png --plan runs/probes/probe_plan.json \
  --out runs/stability/candidates.json
# 3. 从稳定候选选首个实际二值输入闭环通过者，保存条件图/Mask/reference
python -m canny_geometry select --canny E.png --candidates runs/stability/candidates.json \
  --mode st2 --out runs/encoded
# 4. 用未参与选择的prompt/seed评价，均衡测试全部消息状态
python -m canny_geometry generate --conditions runs/encoded \
  --prompt 'A detailed documentary image of a bus on a city road.' --seed 2001 \
  --checkpoint models/control_sd15_canny.pth --clip models/clip-vit-large-patch14 \
  --device cuda --out runs/evaluation
# 5. 接收器仅使用参考和RGB，不传真实消息
python -m canny_geometry decode --reference runs/encoded/reference.json \
  --rgb runs/evaluation/01.png --out runs/decoded_01.json
```

手头已有六张clean RGB时可直接运行第2步；plan需含六个不同`rgb`路径及三个`prompt_group`，最好同时提供prompt、seed与`hint_sha256`。RGB相对路径按plan所在目录解析。旧schema只给RGB/group时无法核验真实生成条件，调用者必须保证来自同一个未改E。所有探测必须同坐标同尺寸；本版没有自动配准。

`candidates_stable_edges.png`保存稳定原边缘支持；`candidates_maps.npz`含support_count、probability、displacement_dispersion_px、source_edge。稳定图只含E上的筛选像素；提取曲线后才做整段复核。`select`在原始完整E上检查编码可写性并建立编辑Mask，编辑Mask可含邻近背景。无载体时明确失败，不回退到未筛候选。先冻结选择/阈值，再生成独立评价图；无可写载体的内容必须计入最终可编码率。

`select`未提供RGB信道校准，reference明确标注输入域诊断；不能据此保证新RGB的消息准确率。低层`embed`可显式传`--calibration`复用历史4A模型，但该模型不保证适用于新筛选分布。

## 重跑随包历史GPU单例

```bash
python -m canny_geometry generate --conditions examples/conditions \
  --prompt 'A realistic documentary photograph of the following scene: A young girl talking on a cell phone while wearing a jacket.' \
  --seed 2026114101 --checkpoint models/control_sd15_canny.pth \
  --clip models/clip-vit-large-patch14 --device cuda --out runs/historical
python -m canny_geometry decode --reference examples/reference_1bit.json \
  --rgb runs/historical/bit1.png --out runs/historical_bit1.json
```

配置固定为DDIM20、CFG5、eta0、control1、float32、空negative prompt；各消息同seed用同一CPU初始噪声。不同权重或数值环境不能保证PNG逐字节相同。

## 可选MAT普通修复

安装`requirements-mat.txt`；解压Release中的领域目录并保留原`protocol.json`字节。根据`thresholds.json`取对应kind/structure的冻结阈值，下面的`THRESHOLD`需要替换为该数值。

```bash
python -m canny_geometry.mat_repair --canny E.png --mask MASK.png \
  --checkpoint ASSET/checkpoints/canny_native.pt --protocol ASSET/protocol.json \
  --structure native --threshold THRESHOLD --device cuda --out runs/mat_repair
```

白色Mask代表缺口；输出`repaired.png`保持原Canny孔外像素不变。PiDiNet-soft需配`--converter ASSET/checkpoints/converter.pt`，选`pidi_soft_native.pt`或`pidi_soft_refined.pt`并匹配structure；先擦除Canny再转换，再擦除转换后缺口。此入口是普通修复诊断，无消息约束W+优化/投影，不等同旧MAT整批实验。概率图与阈值记录供后续分析。

## 为什么历史整批不能直接复现

公开模型解决网络权重；COCO解决源照片。旧批次还依赖其他未随代码交付的**实验状态**：

1. 完整有序内容划分、候选selection与固定samples、ROI/letterbox信息、实际条件图及Mask；normal总体清单不等于某批最终入选名单。
2. 每条曲线实际二值输入拟合得到的码本、Cal信道模型、具体接收器版本，以及各消息/视图的生成计划与噪声。
3. 多层父实验产物及冻结哈希。旧`broad_peak_runtime.py`在导入阶段读取`outputs/canny_curve_stage1_train32_20261110/protocol.json`等；只有同名报告摘录不能充当原文件。
4. 全部历史RGB/观测，或足以逐个重新生成它们的完整输入闭包；完整质量评价还需LPIPS及匹配clean图。
5. MAT从零重训另需训练划分、掩膜、教师及训练流程；四个推理checkpoint不包含这些。

此前包“保留原脚本”只做到源码审阅，不等于可移植复现，连旧脚本`--help`也可能被导入依赖阻断。本版移除这些不能独立运行的入口，留下可运行主线并明确边界。`metrics`只是从已存判决重算75%，`demo`只重读一组三张RGB，均不等于重跑Fresh24/Eval12整批。

原工作区仍保留全部现存资产，日后若要逐批复现应分别导出最小输入闭包、路径无关计划和对应接收器，再逐图重跑比对。新“区域先筛选”需另立协议、全新内容、无载体计失败、独立prompt/seed和重新校准，不能用旧数字作新流程结果。
