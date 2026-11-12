from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/condition_carrier_screen_20261106'
ASSETS = ROOT / 'models/carrier_screen'
MODES = ('softedge', 'lineart', 'scribble', 'depth')
LABELS = ('clean', 'bit0', 'bit1')
CONTROLS = dict(zip(MODES, (
    'control_v11p_sd15_softedge.pth', 'control_v11p_sd15_lineart.pth',
    'control_v11p_sd15_scribble.pth', 'control_v11f1p_sd15_depth.pth')))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def setup(device):
    torch.set_num_threads(4)
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; run with GPU access, no silent CPU fallback.')
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB')).copy()


class Extractor:
    """Frozen official networks, local checkpoints, no implicit downloads."""
    def __init__(self, mode, device):
        self.mode, self.device = mode, device
        if mode in ('softedge', 'scribble'):
            from annotator.hed import ControlNetHED_Apache2
            self.net = ControlNetHED_Apache2()
            self.net.load_state_dict(torch.load(ROOT / 'annotator/ckpts/ControlNetHED.pth', map_location='cpu'))
        elif mode == 'lineart':
            from .lineart_network import Generator
            self.net = Generator(3, 1, 3)
            self.net.load_state_dict(torch.load(ASSETS / 'sk_model.pth', map_location='cpu'))
        elif mode == 'depth':
            from annotator.midas.api import MiDaSInference
            self.net = MiDaSInference('dpt_hybrid')
        else:
            raise ValueError(mode)
        self.net = self.net.float().eval().requires_grad_(False).to(device)

    @torch.inference_mode()
    def __call__(self, rgb):
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float().to(self.device)
        if self.mode in ('softedge', 'scribble'):
            logits = [cv2.resize(v[0, 0].cpu().numpy(), rgb.shape[1::-1]) for v in self.net(tensor)]
            value = (1 / (1 + np.exp(-np.mean(logits, axis=0).astype(np.float64)))).astype(np.float32)
            q = np.clip(value * 255, 0, 255).astype(np.uint8)
            if self.mode == 'scribble':
                from annotator.hed import nms
                q = nms(q, 127, 3.0)
                q = np.where(cv2.GaussianBlur(q, (0, 0), 3.0) > 4, 255, 0).astype(np.uint8)
                value = q.astype(np.float32) / 255
        elif self.mode == 'lineart':
            # Model-card fine lineart: input [0,1], white strokes on black background.
            value = 1 - self.net(tensor / 255)[0, 0].cpu().numpy()
            q = 255 - np.clip((1 - value) * 255, 0, 255).astype(np.uint8)
        else:
            raw = self.net(tensor / 127.5 - 1)[0].cpu().numpy()
            span = float(raw.max() - raw.min())
            if span < 1e-8:
                raise RuntimeError('Degenerate MiDaS depth')
            value = (raw - raw.min()) / span
            q = np.clip(value * 255, 0, 255).astype(np.uint8)
        return q, value


def deformation(size=512, amplitude=3.0):
    """Smooth nonrigid inverse map, identity on boundary; same for all carriers."""
    y, x = np.mgrid[:size, :size].astype(np.float32)
    xx, yy = x / (size - 1), y / (size - 1)
    dx = amplitude * np.sin(2 * np.pi * yy) * np.sin(np.pi * xx)**2 * np.sin(np.pi * yy)**2
    dy = amplitude * np.sin(2 * np.pi * xx) * np.sin(np.pi * yy)**2 * np.sin(np.pi * xx)**2
    return x, y, dx.astype(np.float32), dy.astype(np.float32)


def variants(q, mode):
    x, y, dx, dy = deformation(q.shape[0])
    interp = cv2.INTER_NEAREST if mode == 'scribble' else cv2.INTER_LINEAR
    result = {'clean': q}
    for label, sign in (('bit0', -1), ('bit1', 1)):
        result[label] = cv2.remap(q, x + sign * dx, y + sign * dy, interp,
                                  borderMode=cv2.BORDER_REPLICATE)
    return result


def edge_f1(a, b):
    aa, bb = cv2.Canny(a, 100, 200) > 0, cv2.Canny(b, 100, 200) > 0
    if not aa.any() and not bb.any():
        return 1.0
    kernel = np.ones((3, 3), np.uint8)
    ad, bd = cv2.dilate(aa.astype(np.uint8), kernel), cv2.dilate(bb.astype(np.uint8), kernel)
    precision = float(ad[bb].mean()) if bb.any() else 0.0
    recall = float(bd[aa].mean()) if aa.any() else 0.0
    return 2 * precision * recall / max(precision + recall, 1e-12)


def pixel_metrics(a, b):
    from skimage.metrics import structural_similarity
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64))**2))
    return dict(mse=mse, psnr_db=float(10 * np.log10(255**2 / mse)) if mse else None,
                psnr_infinite=mse == 0,
                ssim=float(structural_similarity(a, b, data_range=255, channel_axis=2)),
                edge_f1_3x3=edge_f1(a, b), changed_pixels=int(np.any(a != b, axis=2).sum()))


def fit_head(features, targets):
    """Fixed ridge head; fit only development contents, no hyperparameter search."""
    x = np.asarray(features, np.float64)
    mean, scale = x.mean(0), x.std(0)
    scale = np.maximum(scale, 0.05)  # Fixed floor for normalized backbone features.
    z = (x - mean) / scale
    onehot = np.eye(3)[targets]
    # Dual form keeps the fit at 6x6, independent of feature dimension.
    weight = z.T @ np.linalg.solve(z @ z.T + np.eye(len(z)), onehot - 1 / 3)
    return dict(mean=mean.tolist(), scale=scale.tolist(), weight=weight.tolist(),
                bias=[1 / 3] * 3, classes=list(LABELS), ridge_lambda=1.0)


def predict(head, feature):
    scores = ((np.asarray(feature) - head['mean']) / head['scale']) @ np.asarray(head['weight']) + head['bias']
    return LABELS[int(np.argmax(scores))], scores.tolist()


class RGBFeatures:
    """Identical frozen ImageNet ResNet50 receiver for every carrier."""
    def __init__(self, device):
        from torchvision.models import resnet50
        net = resnet50(weights=None)
        net.load_state_dict(torch.load(ROOT / 'models/resnet50-11ad3fa6.pth', map_location='cpu'))
        self.net = torch.nn.Sequential(*list(net.children())[:-2]).eval().requires_grad_(False).to(device)
        self.device = device

    @torch.inference_mode()
    def __call__(self, rgb):
        # Preserve geometry: direct square 224 resize, no random crop/augmentation.
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(self.device) / 255
        mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
        std = x.new_tensor([.229, .224, .225])[None, :, None, None]
        x = self.net((x - mean) / std)
        # Retain spatial information instead of only global semantic pooling.
        x = torch.nn.functional.adaptive_avg_pool2d(x, (2, 2)).flatten().cpu().numpy().astype(np.float64)
        return x / max(np.linalg.norm(x), 1e-12) * np.sqrt(x.size)


def condition_features(q, mode):
    x = q.astype(np.float64) / 255
    if mode == 'lineart':
        x = 1 - x
    x = cv2.resize(x, (8, 8), interpolation=cv2.INTER_AREA)
    x = (x - x.mean()) / max(x.std(), .01)
    return x.flatten()
