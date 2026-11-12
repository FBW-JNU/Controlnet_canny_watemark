"""Load the pinned official MAT architecture without changing upstream files."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor'))
sys.path.insert(0, str(ROOT / 'MAT'))


def configure_reference_ops():
    # No nvcc on this host. These are MAT's own reference implementations,
    # run as ordinary PyTorch operations on the selected CPU/GPU.
    from torch_utils.ops import bias_act, upfirdn2d
    bias_act._inited = True
    bias_act._plugin = None
    upfirdn2d._inited = True
    upfirdn2d._plugin = None


def load_mat(checkpoint, device='cpu', channels=3, resolution=512):
    from networks.mat import Generator
    configure_reference_ops()
    # IOPaint's released generator omits noise-strength/constant tensors.
    # Match that explicit supported architecture option, never ignore keys.
    model = Generator(z_dim=512, c_dim=0, w_dim=512,
                      img_resolution=resolution, img_channels=channels,
                      synthesis_kwargs={'use_noise': False})
    state = torch.load(checkpoint, map_location='cpu')
    if channels != 3 or resolution != 512:
        raise ValueError('Pretrained strict load is only defined for RGB MAT-512')
    compatible = model.load_state_dict(state, strict=True)
    assert not compatible.missing_keys and not compatible.unexpected_keys
    return model.eval().requires_grad_(False).to(device)


def mat_forward(model, image, missing, seed=240, return_stg1=False):
    """image is [B,3,H,W] in [0,1], missing is bool [B,1,H,W]."""
    if image.shape[0] != 1 or missing.dtype != torch.bool:
        raise ValueError('Use one image at a time and an explicit missing bool mask')
    torch.manual_seed(seed)
    if image.is_cuda:
        torch.cuda.manual_seed_all(seed)
    # Original generate_image.py uses NumPy float64 z; model mapping converts
    # it to float32. Use the same seeded NumPy distribution here.
    import numpy as np
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, model.z_dim)).to(image.device)
    labels = torch.zeros([1, model.c_dim], device=image.device)
    known = (~missing).float()
    # Explicitly erase missing pixels before model entry, beyond upstream's
    # internal multiplication by known; concealed truth cannot leak.
    masked = (image * 2 - 1) * known
    result = model(masked, known, z, labels, noise_mode='const', return_stg1=return_stg1)
    if return_stg1:
        return tuple((part + 1) / 2 for part in result)
    return (result + 1) / 2
