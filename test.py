import torch

latents = torch.zeros(1, 4, 64, 64)
noise_offset = 0.01
random_noise = torch.randn(latents.shape[0], latents.shape[1], 1, 1)
offset = noise_offset * random_noise
print(offset)