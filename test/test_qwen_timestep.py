"""
Full script that:
  1. Builds your custom exponential-shift σ schedule (1 000 steps, 1 → 0).
  2. Loads the diffusers Flow-Match Euler scheduler and extracts its σ schedule
     at the same 1 000 relative timesteps.
  3. Plots:
        – the custom schedule (smooth line)
        – the diffusers schedule (red scatter)
        – the diffusers schedule inverted vertically: σ → 1 − σ (teal scatter)
"""

import torch
import math
import matplotlib.pyplot as plt
from diffusers import FlowMatchEulerDiscreteScheduler
import random 

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


# ------------------------------------------------------------------
# 1. Custom exponential-shift schedule (1000 steps, 1 → 0)
# ------------------------------------------------------------------
sigma_min, sigma_max, denoising_strength = 0, 1, 1
mu = 0.8
sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
custom_sigmas = torch.linspace(sigma_start, 0, 1000)
custom_sigmas = math.exp(mu) / (math.exp(mu) + (1 / custom_sigmas - 1))

# ------------------------------------------------------------------
# 2. Diffusers schedule at the same 1 000 relative timesteps
# ------------------------------------------------------------------
model_path = r"F:\T2ITrainer\qwen_models\qwen_image_nf4"
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    model_path, subfolder="scheduler"
)

# sample 1 000 relative timesteps
u = compute_density_for_timestep_sampling(
    weighting_scheme="logit_normal",
    batch_size=1000,
    logit_mean=0,
    logit_std=1,
    mode_scale=1.29,
)
indices = (u * scheduler.config.num_train_timesteps).long().clamp(
    0, scheduler.config.num_train_timesteps - 1
)
timesteps = scheduler.timesteps[indices]
device = torch.device("cuda")


def get_sigmas(timesteps, n_dim=1, dtype=torch.float32):
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


diffusers_sigmas = get_sigmas(timesteps)

# ------------------------------------------------------------------
# 3. Invert the diffusers schedule (vertical flip: σ → 1-σ)
# ------------------------------------------------------------------
inverted_diffusers_sigmas = 1.0 - diffusers_sigmas
inverted_diffusers_sigmas = math.exp(mu) / (math.exp(mu) + (1 / inverted_diffusers_sigmas - 1))
# ------------------------------------------------------------------
# 4. Plot everything
# ------------------------------------------------------------------
plt.figure(figsize=(8, 4))

# custom schedule (smooth line)
plt.plot(
    torch.linspace(0, 1, len(custom_sigmas)),
    custom_sigmas.numpy(),
    label="Custom exponential-shift schedule",
    linewidth=1.5,
)

# diffusers schedule (red scatter)
rel_pos = timesteps.float() / scheduler.config.num_train_timesteps
plt.scatter(
    rel_pos.cpu(),
    diffusers_sigmas.cpu(),
    color="crimson",
    s=8,
    label="Diffusers schedule",
    zorder=5,
)

# inverted diffusers schedule (teal scatter)
plt.scatter(
    rel_pos.cpu(),
    inverted_diffusers_sigmas.cpu(),
    color="teal",
    s=8,
    label="Diffusers schedule (inverted)",
    zorder=5,
)

plt.title("Custom vs. Diffusers σ schedule")
plt.xlabel("Relative time (0 = start, 1 = end)")
plt.ylabel("Sigma")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()