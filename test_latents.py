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

vae = AutoencoderKL.from_pretrained(
    "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers",
    subfolder="vae",
    revision=None,
    variant=None,
)
vae.to("cuda").to(torch.float16)
npz_path = "F:/ImageSet/handpick_high_quality_b2_train/1_tags/ganyu_5672750_centered.nphy"
image_path = "F:/ImageSet/handpick_high_quality_b2_train/1_tags/ganyu_5672750_centered.webp"

latents = torch.load(npz_path)
latent = latents['latent'].to("cuda").unsqueeze(0).to(torch.float16)
image_processor = VaeImageProcessor(vae_scale_factor=vae.config.scaling_factor)
with torch.no_grad():
    image = vae.decode(latent / vae.config.scaling_factor, return_dict=False)[0]
image = image_processor.postprocess(image, output_type="pil")[0]
image.save("cat.png")