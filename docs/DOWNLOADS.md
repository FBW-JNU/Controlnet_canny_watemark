# 权重、COCO 与实验资产下载

以下命令从**本代码包根目录**运行。CPU `demo` 和 `metrics` 无需任何模型或完整数据集；新 RGB 生成需要 ControlNet 和 CLIP；恢复批量内容还需 COCO。MAT 是可选分支。

代码 ZIP 不内置数 GB 的公开权重或完整数据集。下载链接来自官方发布页或本机保留的上游实现。后续检查确认COCO主数据/标注、MAT基础和领域权重地址响应HTTP200；HuggingFace请求连接被重置，未判断其失效，也未重新下载大文件。机器可读的来源、文件大小和已知校验值见 [external_assets.json](../provenance/external_assets.json)。

## 1. 当前实际使用的 ControlNet

- 官方仓库：https://huggingface.co/lllyasviel/ControlNet
- 文件直链：https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth
- 本地文件：`models/control_sd15_canny.pth`
- 精确大小：5,710,753,329 bytes；SHA-256：`4de384b16bc2d7a1fb258ca0cbd941d7dd0a721ae996aff89f905299d6923f45`。

```bash
mkdir -p models
curl -fL --retry 3 -C - \
  'https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth' \
  -o models/control_sd15_canny.pth
echo '4de384b16bc2d7a1fb258ca0cbd941d7dd0a721ae996aff89f905299d6923f45  models/control_sd15_canny.pth' | sha256sum -c -
```

这是当前加载器需要的**完整 SD1.5 + Canny ControlNet checkpoint**，包含主干及VAE等权重。无需再下载独立 SD1.5 checkpoint。Diffusers 格式目录、ControlNet v1.1、SDXL 或仅 ControlNet 增量权重不是这里的直接替代品。Canny 使用 OpenCV，不需要下载边缘检测器模型。

## 2. CLIP 文本模型与 tokenizer

- 官方模型：https://huggingface.co/openai/clip-vit-large-patch14
- 本机来源 Git revision：`32bd64288804d66eefd0ccbe215aa642df71cc41`。
- 固定版本入口：https://huggingface.co/openai/clip-vit-large-patch14/tree/32bd64288804d66eefd0ccbe215aa642df71cc41

为匹配 transformers 4.19.2，下载 PyTorch `.bin` 权重及配置/tokenizer，无需重复下载 TensorFlow、Flax 和 safetensors 格式：

```bash
mkdir -p models/clip-vit-large-patch14
for clip_file in config.json merges.txt vocab.json tokenizer.json tokenizer_config.json special_tokens_map.json preprocessor_config.json pytorch_model.bin; do
  curl -fL --retry 3 -C - \
    "https://huggingface.co/openai/clip-vit-large-patch14/resolve/32bd64288804d66eefd0ccbe215aa642df71cc41/${clip_file}" \
    -o "models/clip-vit-large-patch14/${clip_file}" || exit 1
done
echo 'f1a17cdbe0f36fec524f5cafb1c261ea3bbbc13e346e0f74fc9eb0460dedd0d3  models/clip-vit-large-patch14/pytorch_model.bin' | sha256sum -c -
```

权重大小1,710,671,599 bytes，上述哈希来自本机CLIP仓库该revision的Git LFS指针；小配置文件的本地哈希在资产清单中。`generate` 需传 `--clip models/clip-vit-large-patch14`，不能只给 `.bin` 文件。

## 3. COCO 2017 图像与标注

官方索引：https://cocodataset.org/#download

| 下载 | 用途 | 解压后的关键路径 |
|---|---|---|
| [train2017.zip](http://images.cocodataset.org/zips/train2017.zip)，约18GB | 当前曲线主线来源；118,287张 | `datasets/coco/train2017/*.jpg` |
| [annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)，约241MB | captions提供prompt；instances用于部分MAT准备 | `datasets/coco/annotations/captions_train2017.json`、`instances_train2017.json`等 |
| [val2017.zip](http://images.cocodataset.org/zips/val2017.zip)，约1GB，可选 | 部分MAT历史数据准备使用train/val两份instances及图像 | `datasets/coco/val2017/*.jpg` |

```bash
mkdir -p datasets/downloads datasets/coco
curl -fL --retry 3 -C - 'http://images.cocodataset.org/zips/train2017.zip' \
  -o datasets/downloads/train2017.zip
curl -fL --retry 3 -C - 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip' \
  -o datasets/downloads/annotations_trainval2017.zip
unzip -n datasets/downloads/train2017.zip -d datasets/coco
unzip -n datasets/downloads/annotations_trainval2017.zip -d datasets/coco

# 仅在相关分支需要val图像时执行：
curl -fL --retry 3 -C - 'http://images.cocodataset.org/zips/val2017.zip' \
  -o datasets/downloads/val2017.zip
unzip -n datasets/downloads/val2017.zip -d datasets/coco
```

上列直链沿用COCO官方HTTP地址；如网络策略拦截HTTP，从官方索引选择可访问的下载入口。不用test2017替代train2017。数据许可见https://cocodataset.org/#termsofuse 。未取得官方压缩包可靠校验值的项在资产JSON中保留null，不提供猜测哈希。

### 恢复我们实际使用的 normal 内容清单

下载整个COCO不等于恢复本研究抽样总体。本包额外包含：

- [normal_manifest.csv.gz](../data/coco/normal_manifest.csv.gz)：94,625行，顺序、图像ID、prompt和条件统计均来自原清单；`image_path`改为`train2017/<file_name>`。
- [filter_config.json](../data/coco/filter_config.json)：原筛选配置的路径可移植副本。
- [manifest_metadata.json](../data/coco/manifest_metadata.json)：原清单SHA-256、字段、行数及迁移规则。

```bash
python tools/restore_coco_manifest.py --coco-root datasets/coco \
  --out coco/condition_validity/train2017/canny/normal_manifest.csv --check-images
```

此命令按当前COCO目录填写绝对图像路径，保持行序和prompt不变，默认拒绝覆盖。`--check-images`检查所有清单图像存在；省略可先准备清单。原始清单SHA-256为`b0a4cc60ed07fd9f27afd847b47a1917b03c186c93babb92e0a1f4909215abdf`，路径迁移后CSV哈希会变化，需重新记录协议，不可声称满足旧绝对路径文件的哈希冻结。

几何轻量入口直接接受Canny PNG，无需全量COCO。原主实验预处理是RGB长边等比缩至512、Lanczos、灰127居中补边，再`cv2.Canny(RGB,100,200)`；每个候选还受有效ROI等约束。normal条件筛选统计不应替代主实验的这段预处理。

## 4. MAT 的公开基础权重

当前 `mat_common.py` / `mat_canny_model.py` 实际读取的是 **IOPaint 发布的PyTorch生成器**：

- 下载：https://github.com/Sanster/models/releases/download/add_mat/Places_512_FullData_G.pth
- 来源依据：本工作区保留的IOPaint实现`fuxian_papers/records/mat_20261101/iopaint_mat_reference.py`中的`MAT_MODEL_URL`，非推测文件名。
- SHA-256：`0512e37ebba3986b0355130b2e2c1f95736d0778ac82e91b1212b4b21c231312`。
- 上游实现还记录MD5：`8ca927835fa3f5e21d65ffcb165377ed`。

```bash
mkdir -p fuxian_papers/checkpoints/mat
curl -fL --retry 3 -C - \
  'https://github.com/Sanster/models/releases/download/add_mat/Places_512_FullData_G.pth' \
  -o fuxian_papers/checkpoints/mat/Places_512_FullData_G.pth
echo '0512e37ebba3986b0355130b2e2c1f95736d0778ac82e91b1212b4b21c231312  fuxian_papers/checkpoints/mat/Places_512_FullData_G.pth' | sha256sum -c -
```

原作者代码：https://github.com/fenglinglwb/MAT 。原作者模型入口见https://github.com/fenglinglwb/MAT#quick-test ，原始OneDrive链接保存在随包`vendor/MAT/README.md`。原生`.pkl`与上述`.pth`不等价，不可只改文件扩展名；当前架构显式`use_noise=False`并严格加载state_dict。

## 5. 本地训练 MAT 领域权重：单独资产包

这些是我们的领域训练产物，**已发布到 [本项目mat-weights Release](https://github.com/FBW-JNU/Controlnet_canny_watemark/releases/tag/mat-weights)**。公开Places权重不能代替它们重现已报告的领域模型结果。

```bash
mkdir -p datasets/downloads
curl -fL --retry 3 -C - \
  'https://github.com/FBW-JNU/Controlnet_canny_watemark/releases/download/mat-weights/canny-geometry-mat-domain-weights.zip' \
  -o datasets/downloads/canny-geometry-mat-domain-weights.zip
echo '7f7f3f9a09af9dc0061f89ca7a71d9984d87b5ab34bca9913ede255667374858  datasets/downloads/canny-geometry-mat-domain-weights.zip' | sha256sum -c -
```

线上ZIP为694,302,888 bytes，GitHub API报告的SHA-256与本地权重包一致。本地原始文件名为：

```text
canny-geometry-mat-domain-weights_20261112.zip
```

包含实际消息分支使用的4个权重和原始`protocol.json`、`thresholds.json`、MAT许可证、逐文件SHA-256。不包含其他未用于当前消息分支的领域模型，也不包含训练RGB/教师数据。

| 文件 | 大小 | 用途 |
|---|---:|---|
| `canny_native.pt` | 250,569,470 bytes | binary MAT V2分支 |
| `pidi_soft_native.pt` | 250,571,402 bytes | soft native分支 |
| `pidi_soft_refined.pt` | 253,839,597 bytes | soft refined分支 |
| `converter.pt` | 3,264,927 bytes | Canny→PiDiNet-soft转换 |

将资产ZIP解压到独立临时目录后，运行其中的`verify_assets.py`，再把`mat_hed_pidi_repair_20261108/`整个目录复制到原脚本需要的：

```text
fuxian_papers/results/MAE_MAT_related/mat_hed_pidi_repair_20261108/
  checkpoints/{canny_native.pt,pidi_soft_native.pt,pidi_soft_refined.pt,converter.pt}
  protocol.json
  thresholds.json
```

**原始协议字节必须保留**：`converter.pt`会核对训练协议哈希。只迁移资产目录的外部定位；不要把该协议内的历史路径改写后仍按原checkpoint校验。领域权重已作为GitHub Releases附件发布，不要将数百MB checkpoint直接commit进普通Git仓库。

若要从零重新训练这些模型，还需原始数据划分、掩膜与教师产物，以及HED/PiDiNet教师模型和提取代码；当前包只解决当前消息推理分支的权重交付，不是所有历史训练资产的镜像。

## 6. 下载后先运行一次配对生成

安装环境见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)，再运行：

```bash
python -m canny_geometry generate --conditions examples/conditions \
  --prompt 'A realistic documentary photograph of the following scene: A young girl talking on a cell phone while wearing a jacket.' \
  --seed 2026114101 --checkpoint models/control_sd15_canny.pth \
  --clip models/clip-vit-large-patch14 --device cuda --out runs/generated
python -m canny_geometry decode --reference examples/reference_1bit.json \
  --rgb runs/generated/bit1.png --out runs/generated_bit1.json
```

这个命令已在独立GitHub源码快照目录、现有环境及本地权重下执行GPU验证：随包1bit的三张PNG及解码与历史示例一致。没有验证全新环境安装或完整批量性能。完全重跑历史冻结脚本仍需恢复其旧输出依赖、目录与ROI；COCO和权重下载齐全不代表所有旧实验能直接一键重放。

## 7. 完整质量评价所需 LPIPS / AlexNet

`demo`、几何解码与普通ControlNet生成不需要LPIPS。历史报告中的质量四项联合门槛使用`lpips.LPIPS(net='alex', version='0.1')`，除了安装`lpips==0.1.4`外，还需要torchvision的预训练AlexNet主干：

- 官方直链：https://download.pytorch.org/models/alexnet-owt-7be5be79.pth
- 默认缓存通常为`~/.cache/torch/hub/checkpoints/`；也可显式设置`TORCH_HOME`。
- LPIPS自己的`weights/v0.1/alex.pth`随pip包提供，和这个AlexNet主干不是同一文件。

```bash
export TORCH_HOME="$PWD/runs/torch-cache"
mkdir -p "$TORCH_HOME/hub/checkpoints"
curl -fL --retry 3 -C - \
  'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth' \
  -o "$TORCH_HOME/hub/checkpoints/alexnet-owt-7be5be79.pth"
```

URL和文件名来自当前torchvision的AlexNet权重定义；本次未重新下载此权重、未重跑全部质量评估。HED/PiDiNet教师权重和完整训练数据仍需额外交付才能从零重训MAT，当前声明的复现范围不包括该层级。

## 精简版说明

本版普通MAT推理入口为`python -m canny_geometry.mat_repair`，直接加载领域checkpoint，无需额外Places初始化权重。Canny native冻结阈值为0.1，PiDiNet soft native/refined为0.45000000000000007；协议与structure严格核对。消息约束W+旧实验未移植。以上原目录布局仅供历史定位，精简CLI可显式指定资产任意解压位置，见[复现文档](REPRODUCIBILITY.md)。历史整批结果仍缺实验状态，公开下载地址齐全不等于整批可复现。
