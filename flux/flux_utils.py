import math
import random
import torch
from typing import Dict, List, Literal, Optional, Union

from diffusers.utils import USE_PEFT_BACKEND
from diffusers.utils.peft_utils import set_weights_and_activate_adapters

_SET_ADAPTER_SCALE_FN_MAPPING = {
    "MaskedFluxTransformer2DModel": lambda model_cls, weights: weights,
}

# https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py
def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    """
    Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting

# https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py#L236
def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None, 
    pos_logit_mean: float = None, pos_selection_lambda: float = 0.5
):
    """
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    elif weighting_scheme == "logit_snr":
        logsnr = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        # from https://arxiv.org/pdf/2411.14793
        u = 1 - torch.nn.functional.sigmoid(-logsnr/2)
    elif weighting_scheme == "double_logit_normal":
        u1 = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u2 = torch.normal(mean=pos_logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        
        if random.random() > pos_selection_lambda:
            u = u1
        else:
            u = u2
        u = torch.nn.functional.sigmoid(u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u

def set_adapters(
        transformer_model,
        adapter_names: Union[List[str], str],
        weights: Optional[Union[float, Dict, List[float], List[Dict], List[None]]] = None,
    ):
    """
    Set the currently active adapters for use in the UNet.

    Args:
        adapter_names (`List[str]` or `str`):
            The names of the adapters to use.
        adapter_weights (`Union[List[float], float]`, *optional*):
            The adapter(s) weights to use with the UNet. If `None`, the weights are set to `1.0` for all the
            adapters.

    Example:

    ```py
    from diffusers import AutoPipelineForText2Image
    import torch

    pipeline = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
    ).to("cuda")
    pipeline.load_lora_weights(
        "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
    )
    pipeline.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
    pipeline.set_adapters(["cinematic", "pixel"], adapter_weights=[0.5, 0.5])
    ```
    """
    if not USE_PEFT_BACKEND:
        raise ValueError("PEFT backend is required for `set_adapters()`.")

    adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names

    # Expand weights into a list, one entry per adapter
    # examples for e.g. 2 adapters:  [{...}, 7] -> [7,7] ; None -> [None, None]
    if not isinstance(weights, list):
        weights = [weights] * len(adapter_names)

    if len(adapter_names) != len(weights):
        raise ValueError(
            f"Length of adapter names {len(adapter_names)} is not equal to the length of their weights {len(weights)}."
        )

    # Set None values to default of 1.0
    # e.g. [{...}, 7] -> [{...}, 7] ; [None, None] -> [1.0, 1.0]
    weights = [w if w is not None else 1.0 for w in weights]

    # e.g. [{...}, 7] -> [{expanded dict...}, 7]
    scale_expansion_fn = _SET_ADAPTER_SCALE_FN_MAPPING[transformer_model.__class__.__name__]
    weights = scale_expansion_fn(transformer_model, weights)

    set_weights_and_activate_adapters(transformer_model, adapter_names, weights)