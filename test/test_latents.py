from diffusers import (
    AutoencoderKL,
)
from diffusers.image_processor import VaeImageProcessor
import torch
from PIL import Image
import numpy

from torchvision import transforms
import cv2
import torchvision.transforms as T
import torchvision

vae_path = "F:/models/VAE/sdxl_vae.safetensors"
vae = AutoencoderKL.from_single_file(
    vae_path
)


vae.to("cuda").to(torch.float16)
npz_path = "F:/ImageSet/kolors_slider_anime/positive/1.nplatent"
# image_path = "alan-w-ZpmFJoWRqUE-unsplash.webp"

latents = torch.load(npz_path)
latent = latents['latent'].to("cuda").unsqueeze(0).to(torch.float16)
image_processor = VaeImageProcessor(vae_scale_factor=vae.config.scaling_factor)
with torch.no_grad():
    image = vae.decode(latent / vae.config.scaling_factor, return_dict=False)[0]
image = image_processor.postprocess(image, output_type="pil")[0]
image.save("bird.png")