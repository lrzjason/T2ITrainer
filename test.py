import torch

cache_path = "F:/ImageSet/ObjectRemoval/temp/I-210618_I01001_W01_F0179_M_masked_image.npfluxlatent"
latents = torch.load(cache_path, weights_only=True)["latent"]
print(latents.shape)
