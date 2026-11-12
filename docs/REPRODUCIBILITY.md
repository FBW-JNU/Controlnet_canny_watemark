# 操作与复现范围

## 环境与历史示例

建议Python3.9。精确实测版本见[reproduction_environment.json](../provenance/reproduction_environment.json)。原兼容环境为torch2.0.1+cu118、torchvision0.15.2、numpy1.24.4、scipy1.13.1、OpenCV4.11、Pillow11.3。未验证空环境在线安装。

```bash
# 新建环境时建议固定pip；旧Lightning元数据不兼容部分新版pip解析器
conda create -n canny-repro python=3.9 pip=24.0 -y
conda activate canny-repro
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

## 整批复现：历史冻结评测

本次已补齐历史批次的最小输入闭包：291个去重Canny/Mask、54个参考、每图prompt/seed、4A/4B接收器及冻结模型、配对质量计划。`benchmarks/history/manifest.json`只用相对路径；无需历史`outputs`或原照片目录。预期PNG哈希/判决用于结果比对，不参与生成或读码。

安装生成环境及`requirements-evaluation.txt`，按[DOWNLOADS](DOWNLOADS.md)准备ControlNet、CLIP和LPIPS AlexNet缓存后：

```bash
python -m pip install -r requirements-evaluation.txt
python tools/check_benchmark.py
python -m canny_geometry.benchmark verify --manifest benchmarks/history/manifest.json
# 单GPU用 --gpus 0；两GPU用 --gpus 0,1
python tools/run_reproduction.py history --gpus 0,1 \
  --checkpoint models/control_sd15_canny.pth --clip models/clip-vit-large-patch14 \
  --out runs/history
```

该命令重建编码输入、重生成748张RGB、独立解码732行、从108张Cal图重建4A Auto校准、复算576条画质记录。732逻辑行包含clean、轴状态以及共享图像的两策略记录；消息恢复率仅以有真实消息的行计分。日志与RGB保留在指定`runs`目录，可断点续跑；更改清单或输出图会使续跑校验失败。

分步读码和质量核对：

```bash
python -m canny_geometry.benchmark decode --manifest benchmarks/history/manifest.json \
  --rgb-root runs/history/generated --out runs/history/decoded
python -m canny_geometry.benchmark quality --manifest benchmarks/history/manifest.json \
  --rgb-root runs/history/generated --out runs/history/quality
```

本范围重放**冻结选择后的消息与质量评测**；历史候选发现、所有早期诊断、4B训练和MAT训练不属于该命令。4A Auto校准另从Cal观测重建，不使用Eval拟合。质量清单保留原来的全部单轴与四码字条目，不能把它们与只数四码字的历史摘要混为同一分母。

## 新流程：从原始Canny完整重建并重复

随包`benchmarks/region_first/protocol.json`固定6个Cal、12个Eval内容，附18张原始Canny；运行本实验不需要下载完整COCO。协议在新探测前冻结，不换掉失败内容。需要研究其他内容时应新建协议，不能用结果修改这个冻结协议。

```bash
python tools/run_reproduction.py pilot --gpus 0,1 \
  --checkpoint models/control_sd15_canny.pth --clip models/clip-vit-large-patch14 \
  --out runs/region_first --replay
```

每轮从原始E生成108张clean探测；完成像素稳定筛选后提曲线，检查实际四码字输入并冻结全部18个选择；生成60张Cal图，单独校准并冻结；再生成120张Eval图、解码及质量评价。`--replay`在独立目录完整重建第二轮，比较288张PNG、曲线samples、码本、编辑输入、校准及全部判决，结果见`replay_comparison.json`。不是复制第一次的RGB。

要与本包公布结果比较，再运行：

```bash
python tools/compare_reproduction.py --run runs/region_first/primary \
  --out runs/region_first/published_comparison.json
```

失败分析可独立重建，不改变选择或校准：

```bash
python tools/diagnose_reproduction.py --run runs/region_first/primary \
  --out runs/region_first/primary/failure_diagnostics.json
```

主结果在`primary/result.json`，画质在`primary/eval_quality/quality.json`，筛选与模型在`primary/{selection,calibration}.json`。校准仅用Cal，不复用旧4A参数；若Cal观测不足，协议规定使用输入域回退并明确记录。Eval同时比较新校准与输入域读码，使用相同图像；这不是source-only对照。

新批次的冻结输入、计划和金标准另附在`benchmarks/region_first/frozen/`，便于只重放消息阶段，例如：

```bash
python -m canny_geometry.benchmark generate \
  --manifest benchmarks/region_first/frozen/eval_manifest.json \
  --checkpoint models/control_sd15_canny.pth --clip models/clip-vit-large-patch14 \
  --out runs/new_frozen
python -m canny_geometry.benchmark decode \
  --manifest benchmarks/region_first/frozen/eval_manifest.json \
  --rgb-root runs/new_frozen --out runs/new_frozen_decoded
```

冻结消息阶段的清单已附金标准PNG/判决；它用于重放，不用于证明筛选顺序。完整顺序验证应使用上面的`pilot --replay`。验收报告见[REPRODUCTION_RESULT](REPRODUCTION_RESULT.md)。

## 仍有边界

- 已验证现有兼容环境与本地权重；没有验证全新环境在线安装。环境差异可能影响PNG逐字节一致，不能承诺所有GPU/版本的哈希相同。
- 冻结评测的可复现，不等于多载体长消息、RM/RS实图纠错、仅Mask读码或MAT收益已经完成。
- 历史内容发现与MAT从零训练仍需更早协议、训练划分、掩膜、教师及训练代码；推理权重不包含这些。
