from mmcv import Registry

from utils.utils import set_grad_checkpoint

from utils.PixArt import PixArt, PixArt_XL_2
from utils.PixArtMS import PixArtMS, PixArtMS_XL_2, PixArtMSBlock

MODELS = Registry('models')
def build_model(cfg, use_grad_checkpoint=False, use_fp32_attention=False, gc_step=1, **kwargs):
    if isinstance(cfg, str):
        cfg = dict(type=cfg)
    model = MODELS.build(cfg, default_args=kwargs)
    if use_grad_checkpoint:
        set_grad_checkpoint(model, use_fp32_attention=use_fp32_attention, gc_step=gc_step)
    return model
