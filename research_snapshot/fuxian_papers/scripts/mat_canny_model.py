"""Original MAT adapted to Canny, without training or extra edge branches."""
import math
import numpy as np
import torch
from mat_common import configure_reference_ops, load_mat, mat_forward


def convert_rgb_state(source, target):
    """Convert RGB weights to native gray; not whole-network equivalence."""
    result, audit = {}, []
    lum = torch.tensor([.299, .587, .114])
    for key, dst in target.items():
        src = source[key]
        if src.shape == dst.shape:
            result[key] = src
            continue
        if key == 'synthesis.first_stage.conv_first.conv.weight':
            value = torch.cat([src[:, :1], src[:, 1:].sum(1, keepdim=True)], 1) * math.sqrt(2/4)
        elif key == 'synthesis.enc.EncConv_Block_512x512.conv0.weight':
            value = torch.cat([src[:, :1], src[:, 1:4].sum(1, keepdim=True),
                               src[:, 4:7].sum(1, keepdim=True)], 1) * math.sqrt(3/7)
        elif '.toRGB.' in key and key.endswith('bias'):
            value = (src * lum).sum().reshape(1)
        elif '.toRGB.' in key and key.endswith('conv.weight'):
            value = (src * lum[None, :, None, None, None]).sum(1, keepdim=True)
        else:
            raise ValueError((key, src.shape, dst.shape))
        assert value.shape == dst.shape
        result[key] = value
        audit.append(dict(key=key, source=list(src.shape), target=list(dst.shape)))
    assert set(source) == set(target)
    return result, audit


def native_mat(weight, device='cpu'):
    from networks.mat import Generator
    configure_reference_ops()
    model = Generator(z_dim=512,c_dim=0,w_dim=512,img_resolution=512,
                      img_channels=1,synthesis_kwargs={'use_noise':False})
    state, audit = convert_rgb_state(torch.load(weight, map_location='cpu'), model.state_dict())
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False).to(device), audit


def load_canny_mat(weight, device='cpu', single_channel=False):
    return native_mat(weight, device)[0] if single_channel else load_mat(weight, device)


@torch.inference_mode()
def repair_canny(model, edge, missing, seed=240):
    """Return final/coarse probabilities; input is binary [1,1,512,512]."""
    if edge.shape != (1, 1, 512, 512) or missing.shape != edge.shape:
        raise ValueError('One 512x512 Canny image and matching mask are required')
    if missing.dtype != torch.bool:
        raise ValueError('The missing mask must be boolean')
    if model.img_channels == 3:
        return mat_forward(model, edge.repeat(1, 3, 1, 1), missing,
                           seed=seed, return_stg1=True)
    torch.manual_seed(seed)
    if edge.is_cuda:
        torch.cuda.manual_seed_all(seed)
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, 512)).to(edge.device)
    known = (~missing).float()
    final, coarse = model((edge * 2 - 1) * known, known, z,
                          torch.zeros(1, 0, device=edge.device),
                          noise_mode='const', return_stg1=True)
    return tuple(torch.where(missing, (raw + 1) / 2, edge)
                 for raw in (final, coarse))
