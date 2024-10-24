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

# vae_path = "F:/models/VAE/sdxl_vae_fp16fix.safetensors"
vae_path = "F:/models/VAE/SD3_16c_vae.safetensors"
# vae = AutoencoderKL.from_single_file(
#     vae_path
# )


vae = AutoencoderKL.from_pretrained(
    "F:/T2ITrainer/sd3.5L",
    subfolder="vae",
)


vae.to("cuda").to(torch.float16)
# npz_path = "F:/ImageSet/vit_train/anatomy/train/good_anatomy/4_prompt_res_1344x1344 - Copy (2).nplatent"
# npz_path = "F:/ImageSet/kolors_cosplay/train/maileji/maileji_1.nplatent"
# npz_path = "F:/ImageSet/kolors_cosplay/train/Azami Onlyfans/Azami - 2B [20P]/01.nplatent"
# image_path = "alan-w-ZpmFJoWRqUE-unsplash.webp"
npz_path = "F:/ImageSet/sd35/diverse-photo-3740/0001.npsd35latent"

latents = torch.load(npz_path)
latent = latents['latent'].to("cuda").unsqueeze(0).to(torch.float16)
image_processor = VaeImageProcessor(vae_scale_factor=vae.config.scaling_factor)
with torch.no_grad():
    image = vae.decode(latent / vae.config.scaling_factor, return_dict=False)[0]
image = image_processor.postprocess(image, output_type="pil")[0]
image.save("test.png")