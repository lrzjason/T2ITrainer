

import torch
import os
from PIL import Image
from diffusers import AutoencoderKLQwenImage
from diffusers.image_processor import VaeImageProcessor
from typing import List, Optional

# Try to import safetensors
try:
    from safetensors import safe_open
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    print("safetensors library not available. Install with: pip install safetensors")
    SAFETENSORS_AVAILABLE = False

@torch.no_grad()
def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pretrained_model_name_or_path = r"F:/T2ITrainer/qwen_models/qwen_image_edit_plus"
    vae = AutoencoderKLQwenImage.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    # vae = vae.to(device, dtype=torch.float32)
    vae.eval()  # Important: disables unnecessary ops like dropout

    vae_scale_factor = 2 ** len(vae.temperal_downsample)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)
    
    latent_path = r"D:\ComfyUI\output\test_00001_.latent"
    latent = load_file(latent_path, device="cpu")
    latents = latent["latent_tensor"]
    latents = latents.to(vae.dtype)
    # latents_mean = (
    #     torch.tensor(vae.config.latents_mean)
    #     .view(1, vae.config.z_dim, 1, 1, 1)
    #     .to(latents.device, latents.dtype)
    # )
    # latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
    #     latents.device, latents.dtype
    # )
    # latents = latents / latents_std + latents_mean
    image = vae.decode(latents, return_dict=False)[0][:, :, 0]
    image = image_processor.postprocess(image, output_type="pil")
    
    output_dir = r"D:\ComfyUI\output\test_vae"
    output_path = os.path.join(output_dir, f"test_vae.png")
    image[0].save(output_path)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()