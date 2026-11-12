"""Run the U-Sketch simplifier and official ControlNet scribble references offline.

Improved Scribble Extraction was identified from the user-provided ETRI page.
ControlNet HED/PIDI are references; improved_scribble_approx.py implements the
abstract's stages separately, with explicitly assumed parameters.
"""
import argparse
import ast
import hashlib
import html
import importlib.util
import json
import platform
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def official_definitions(path, names, namespace):
    """Load exact official definitions without CUDA-only wrappers/GUI imports."""
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def load_models(device, out):
    spec = importlib.util.spec_from_file_location('usketch_official_models', ROOT / 'U-Sketch/models.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    simplifier = module.SketchSimplificationNetwork()
    src_path = ROOT / 'checkpoints/sketch_simplification/model_gan.pth'
    src = torch.load(src_path, map_location='cpu', weights_only=True)
    # Mirror groups the same sequential layers into layer1...layer7. Numeric
    # stage/layer ordering preserves their actual forward order.
    keys = sorted(src, key=lambda k: tuple(int(x) for x in re.findall(r'\d+', k)) + (0 if k.endswith('weight') else 1,))
    dst = simplifier.state_dict()
    assert len(keys) == len(dst) == 46
    mapping = []
    converted = {}
    for source_key, (target_key, target_value) in zip(keys, dst.items()):
        assert src[source_key].shape == target_value.shape, (source_key, target_key)
        converted[target_key] = src[source_key]
        mapping.append({'source': source_key, 'target': target_key, 'shape': list(target_value.shape)})
    simplifier.load_state_dict(converted, strict=True)
    converted_path = ROOT / 'checkpoints/sketch_simplification/usketch_simplifier.pth'
    if not converted_path.exists():
        torch.save(converted, converted_path)
    else:
        previous = torch.load(converted_path, map_location='cpu', weights_only=True)
        assert set(previous) == set(converted)
        assert all(torch.equal(previous[k], converted[k]) for k in converted)
    write_json(out / 'simplifier_key_mapping.json', mapping)

    ns = official_definitions(
        ROOT / 'Improved-Scribble/upstream/annotator/hed/__init__.py',
        ['DoubleConvBlock', 'ControlNetHED_Apache2'], {'torch': torch})
    hed = ns['ControlNetHED_Apache2']()
    hed_path = ROOT / 'checkpoints/improved_scribble/ControlNetHED.pth'
    hed.load_state_dict(torch.load(hed_path, map_location='cpu', weights_only=True), strict=True)

    pidi_path = ROOT / 'Improved-Scribble/upstream/annotator/pidinet/model.py'
    tree = ast.parse(pidi_path.read_text())
    # basicsr.img2tensor is only used by the upstream __main__ demo.
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == 'basicsr.utils')]
    pns = {'__name__': 'official_pidinet_model'}
    exec(compile(tree, str(pidi_path), 'exec'), pns)
    pidi = pns['pidinet']()
    pidi_weight = ROOT / 'checkpoints/improved_scribble/table5_pidinet.pth'
    state = torch.load(pidi_weight, map_location='cpu', weights_only=True)['state_dict']
    pidi.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=True)
    for model in (simplifier, hed, pidi):
        model.to(device).eval().requires_grad_(False)
    util = official_definitions(ROOT / 'Improved-Scribble/upstream/annotator/util.py',
                                ['nms', 'resize_image'], {'cv2': cv2, 'np': np})
    write_json(out / 'weights.json', [
        {'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': digest(p)}
        for p in [src_path, converted_path, hed_path, pidi_weight]])
    return simplifier, hed, pidi, util


@torch.inference_mode()
def simplify(model, gray, device):
    h, w = gray.shape
    x = torch.from_numpy(gray.copy()).to(device=device, dtype=torch.float32)[None, None] / 255
    x = (x - 0.9664114577640158) / 0.0858381272736797
    x = torch.nn.functional.pad(x, (0, -w % 8, 0, -h % 8), mode='replicate')
    pred = model(x).float()[0, 0, :h, :w].cpu().numpy()
    # U-Sketch converts through uint8 by truncating, rather than rounding.
    return (pred * 255).clip(0, 255).astype(np.uint8)


@torch.inference_mode()
def hed_soft(model, rgb, device):
    h, w = rgb.shape[:2]
    x = torch.from_numpy(rgb.copy()).to(device=device, dtype=torch.float32).permute(2, 0, 1)[None]
    projections = model(x)
    edges = [cv2.resize(e.cpu().numpy().astype(np.float32)[0, 0], (w, h), interpolation=cv2.INTER_LINEAR)
             for e in projections]
    logits = np.stack(edges, axis=2).mean(axis=2).astype(np.float64)
    edge = 1 / (1 + np.exp(-logits))
    return (edge * 255).clip(0, 255).astype(np.uint8)


@torch.inference_mode()
def pidi_soft(model, rgb, device):
    x = torch.from_numpy(rgb[:, :, ::-1].copy()).to(device=device, dtype=torch.float32).permute(2, 0, 1)[None] / 255
    return (model(x)[-1].cpu().numpy()[0, 0] * 255).clip(0, 255).astype(np.uint8)


def scribble_postprocess(soft, nms):
    result = nms(soft, 127, 3.0)
    result = cv2.GaussianBlur(result, (0, 0), 3.0)
    result[result > 4] = 255
    result[result < 255] = 0
    return result


def stats(white_paper):
    ink = (white_paper < 128).astype(np.uint8)
    count, _, cc_stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    return {'ink_fraction': float(ink.mean()), 'components': int(count - 1),
            'components_ge16px': int((cc_stats[1:, cv2.CC_STAT_AREA] >= 16).sum())}


def save_row(out, group, name, images, metadata):
    folder = out / group / name
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, array in images.items():
        path = folder / (key + '.png')
        Image.fromarray(array).save(path)
        paths[key] = str(path.relative_to(out))
    record = dict(metadata, group=group, name=name, images=paths,
                  stats={key: stats(arr) for key, arr in images.items() if arr.ndim == 2})
    write_json(folder / 'record.json', record)
    return record


def contact_sheet(out, records, columns, name, thumb=256):
    row_height = thumb + 34
    sheet = Image.new('RGB', (thumb * len(columns), row_height * len(records)), '#ededed')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    for i, record in enumerate(records):
        for j, (key, label) in enumerate(columns):
            img = Image.open(out / record['images'][key]).convert('RGB')
            img.thumbnail((thumb - 4, thumb - 4))
            x, y = j * thumb, i * row_height
            draw.text((x + 4, y + 2), label, fill='black', font=font)
            draw.text((x + 4, y + 18), record['name'], fill='black', font=font)
            sheet.paste(img, (x + (thumb - img.width) // 2, y + 34 + (thumb - img.height) // 2))
    sheet.save(out / name)


def render_html(out, records):
    headers = {
        'official': [('input', '官方输入线稿'), ('simplified', 'U-Sketch 简化网络输出')],
        'coco': [('rgb', 'RGB'), ('canny', 'Canny 100/200'), ('hed', 'ControlNet HED Scribble 参照'),
                 ('pidi', 'ControlNet PIDI Scribble 参照'),
                 ('improved_approx', 'Improved Scribble：摘要近似版'),
                 ('simplified_canny', 'U-Sketch(Canny)'),
                 ('simplified_hed', 'U-Sketch(HED Scribble)'), ('simplified_pidi', 'U-Sketch(PIDI Scribble)')],
    }
    body = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>线稿提取与简化复现</title>',
            '<style>body{font:16px system-ui;margin:24px;color:#222}table{border-collapse:collapse;margin-bottom:35px}td,th{border:1px solid #ccc;padding:8px;vertical-align:top}img{max-width:270px;max-height:360px;display:block}small{display:block;max-width:260px}h2{margin-top:35px}</style>',
            '<h1>线稿提取与 U-Sketch 简化网络：初步推理结果</h1>',
            '<p>U-Sketch 使用官方模型定义、GAN 转换权重与官方归一化参数。所有线稿统一显示为白底黑线，点击图片查看原始输出。</p>',
            '<p><b>Improved Scribble 为摘要近似实现。</b><a href="https://ksp.etri.re.kr/ksp/article/read?id=73493">论文页面</a>未附全文、代码和参数；HED 变体及后处理参数均为本地假设。HED/PIDI 列仅为 ControlNet 官方参照。</p>',
            '<p><a href="main_comparison.jpg">主要对比总览</a> · <a href="improved_stages.jpg">近似版中间阶段</a> · <a href="REPORT_ZH.md">复现说明</a></p>',
            '<p>COCO 为本地既有测试划分按 sample_id 排序的前若干图，无筛选；无配对简化真值，以下墨迹比例和连通域数仅用于描述输出。</p>']
    for group, cols in headers.items():
        body += ['<h2>' + ('官方公开样例' if group == 'official' else 'COCO 自然场景迁移') + '</h2>',
                 '<table><tr><th>样例</th>' + ''.join('<th>' + label + '</th>' for _, label in cols) + '</tr>']
        for rec in records:
            if rec['group'] != group:
                continue
            body.append('<tr><td>' + html.escape(rec['name']) + '<small>' + html.escape(rec.get('caption', '')) + '</small></td>')
            for key, label in cols:
                if key not in rec['images']:
                    body.append('<td>尚未运行摘要近似版</td>')
                    continue
                p = rec['images'][key]
                info = rec['stats'].get(key)
                text = '' if info is None else f"墨迹 {info['ink_fraction']:.2%}；连通域 {info['components']}"
                body.append(f'<td><a href="{p}"><img loading="lazy" src="{p}" alt="{label}"></a><small>{text}</small></td>')
            body.append('</tr>')
        body.append('</table>')
    body.append('</html>')
    (out / 'index.html').write_text('\n'.join(body))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/scribble_usketch_20261108')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--coco-count', type=int, default=12)
    parser.add_argument('--resolution', type=int, default=512)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    write_json(args.out / 'environment.json', {
        'python': sys.version, 'torch': torch.__version__, 'opencv': cv2.__version__,
        'platform': platform.platform(), 'cuda_available': torch.cuda.is_available(),
        'device': args.device, 'threads': args.threads, 'resolution': args.resolution,
        'simplification_scope': 'U-Sketch simplification subnetwork only; no image synthesis',
        'improved_scribble_identity': 'ETRI 73493; abstract-based approximation in a separate script; HED/PIDI are references',
    })
    simplifier, hed, pidi, util = load_models(args.device, args.out)
    records = []
    official = sorted((ROOT / 'U-Sketch/example-sketches').glob('*.png'))
    official += [ROOT / 'Sketch-Simplification/upstream' / p for p in ['test.png', 'figs/fig06_pepper.png', 'figs/fig07_tokage.png']]
    for path in official:
        started = time.monotonic()
        image = Image.open(path).convert('L')
        original_size = image.size
        image.thumbnail((args.resolution, args.resolution), Image.Resampling.LANCZOS)
        gray = np.array(image)
        pred = simplify(simplifier, gray, args.device)
        name = ('usketch_' if 'example-sketches' in str(path) else 'simoserra_') + path.stem
        rec = save_row(args.out, 'official', name, {'input': gray, 'simplified': pred}, {
            'source': str(path.relative_to(ROOT)), 'source_sha256': digest(path),
            'original_size': original_size, 'run_size': image.size,
            'seconds': time.monotonic() - started,
        })
        records.append(rec)
        write_json(args.out / 'samples.json', records)
        print('completed', name, round(rec['seconds'], 2), flush=True)
    data_root = ROOT / 'data/edge_completion_coco256_v1'
    samples = [json.loads(line) for line in (data_root / 'splits/test.jsonl').read_text().splitlines()]
    samples = sorted(samples, key=lambda r: r['sample_id'])[:args.coco_count]
    for sample in samples:
        started = time.monotonic()
        source = Path(sample['source_path'])
        # Preserve the full image and aspect ratio, as in ControlNet's annotator.
        rgb = np.array(ImageOps.exif_transpose(Image.open(source)).convert('RGB'))
        rgb = util['resize_image'](rgb, args.resolution)
        canny = cv2.Canny(rgb, 100, 200)
        soft_hed = hed_soft(hed, rgb, args.device)
        soft_pidi = pidi_soft(pidi, rgb, args.device)
        edge_hed = scribble_postprocess(soft_hed, util['nms'])
        edge_pidi = scribble_postprocess(soft_pidi, util['nms'])
        images = {'rgb': rgb, 'canny': 255 - canny, 'hed': 255 - edge_hed,
                  'pidi': 255 - edge_pidi, 'soft_hed': 255 - soft_hed, 'soft_pidi': 255 - soft_pidi}
        for key in ['canny', 'hed', 'pidi']:
            images['simplified_' + key] = simplify(simplifier, images[key], args.device)
        rec = save_row(args.out, 'coco', sample['sample_id'], images, {
            'source': str(source), 'source_sha256': digest(source),
            'caption': sample.get('caption', ''), 'selection': 'first sample_ids sorted in existing local test split',
            'run_size': list(rgb.shape[1::-1]), 'seconds': time.monotonic() - started,
        })
        records.append(rec)
        write_json(args.out / 'samples.json', records)
        render_html(args.out, records)
        print('completed', sample['sample_id'], round(rec['seconds'], 2), flush=True)
    contact_sheet(args.out, records[:10], [('input', 'Official input'), ('simplified', 'U-Sketch simplified')], 'official_comparison.jpg')
    coco = [r for r in records if r['group'] == 'coco']
    if coco:
        contact_sheet(args.out, coco, [('rgb', 'RGB'), ('canny', 'Canny'), ('hed', 'HED scribble ref.'),
                      ('pidi', 'PIDI scribble ref.'), ('simplified_canny', 'U-Sketch(Canny)'),
                      ('simplified_hed', 'U-Sketch(HED)'), ('simplified_pidi', 'U-Sketch(PIDI)')], 'coco_comparison.jpg')
    render_html(args.out, records)
    print('DONE', args.out, flush=True)


if __name__ == '__main__':
    main()
