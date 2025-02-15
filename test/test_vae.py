from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import cv2
import torchvision.transforms as T
import torchvision

# Load the VAE model
vae_path = "F:/models/VAE/sdxl_vae_fp16fix.safetensors"
vae = AutoencoderKL.from_single_file(vae_path)
vae.to("cuda").to(torch.float16)

image_path = "F:/ImageSet/3dkitten/_00e8a015-44b1-4e25-8c6a-c2c3c42d4b8e.jpg"
image = Image.open(image_path)
img_processor = VaeImageProcessor()
image = img_processor.numpy_to_pt(img_processor.pil_to_numpy(image)).to("cuda").to(torch.float16)

def encode_img(input_img):
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    if len(input_img.shape)<4:
        input_img = input_img.unsqueeze(0)
    with torch.no_grad():
        latent = vae.encode(input_img*2 - 1) # Note scaling
    return 0.18215 * latent.latent_dist.sample()

def decode_img(latents):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach()
    return image

latent = encode_img(image)
image2 = decode_img(latent)

# save image2
image2 = image2.squeeze(0).permute(1, 2, 0).cpu().numpy()
image2 = (image2 * 255).astype(np.uint8)

image2 = Image.fromarray(image2).convert("RGB")
image2.save("test.jpg")
