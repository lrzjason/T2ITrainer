import math
import random
import torch

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
        
        if random.random() < pos_selection_lambda:
            u = u1
        else:
            u = u2
        u = torch.nn.functional.sigmoid(u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u
