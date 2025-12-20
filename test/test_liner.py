import os
import argparse
import torch
import torchvision.transforms as T
from PIL import Image
from diffusers import AutoencoderKLQwenImage
from diffusers.image_processor import VaeImageProcessor
from typing import List, Optional


def decode_vae_latent(vae, image_processor, latents: torch.Tensor):
    latents = latents.to(vae.dtype)
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents = latents / latents_std + latents_mean
    image = vae.decode(latents, return_dict=False)[0][:, :, 0]
    image = image_processor.postprocess(image, output_type="pil")
    return image


def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


@torch.inference_mode()
def encode_vae_image(vae, latent_channels, image: torch.Tensor, generator: torch.Generator):
    if isinstance(generator, list):
        image_latents = [
            retrieve_latents(vae.encode(image[i : i + 1]), generator=generator[i], sample_mode="argmax")
            for i in range(image.shape[0])
        ]
        image_latents = torch.cat(image_latents, dim=0)
    else:
        image_latents = retrieve_latents(vae.encode(image), generator=generator, sample_mode="argmax")
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, latent_channels, 1, 1, 1)
        .to(image_latents.device, image_latents.dtype)
    )
    latents_std = (
        torch.tensor(vae.config.latents_std)
        .view(1, latent_channels, 1, 1, 1)
        .to(image_latents.device, image_latents.dtype)
    )
    image_latents = (image_latents - latents_mean) / latents_std
    return image_latents


def linear_interpolation(
    learning_target: torch.Tensor,      # (B, 16, F, H, W)
    reference_list: torch.Tensor,       # (B, 16, F, H, W)
    reasoning_frame: int,                # e.g., 6
    gamma: float = 0.5
) -> List[torch.Tensor]:
    assert reference_list.shape == learning_target.shape
    assert reasoning_frame >= 0
    B, C, F, H, W = learning_target.shape
    assert F == 1, "Only single-frame interpolation supported"

    ref = reference_list.squeeze(2)      # (B, C, H, W)
    tgt = learning_target.squeeze(2)     # (B, C, H, W)

    total_steps = reasoning_frame + 2
    t = torch.linspace(0.0, 1.0, steps=total_steps, device=ref.device)
    alphas = t ** gamma  # Non-linear spacing biased toward target

    latents = []
    for alpha in alphas:
        interp = (1 - alpha) * tgt + alpha * ref  # (B, C, H, W)
        interp = interp.unsqueeze(2)              # (B, C, 1, H, W)
        latents.append(interp)

    return latents
def resize_with_aspect_ratio(image: Image.Image, target_size: int) -> Image.Image:
    """
    Resize image so that the shorter side = target_size,
    then adjust both dimensions to be divisible by 8.
    """
    # Step 1: Resize shorter side to target_size, preserve aspect ratio
    width, height = image.size
    scale = target_size / min(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)

    # Step 2: Make dimensions divisible by 8 (floor to nearest multiple of 8)
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8

    # Step 3: Center crop to adjusted size
    left = (image.width - new_width) // 2
    top = (image.height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    image = image.crop((left, top, right, bottom))
    
    return image


def load_and_preprocess_image(image_processor, image_path: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = resize_with_aspect_ratio(image, 256)
    image_height, image_width = image_processor.get_default_height_width(image)
    image = image_processor.preprocess(image, image_height, image_width)
    image = image.unsqueeze(2)
    return image


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path_ref", type=str)
    parser.add_argument("--image_path_target", type=str)
    parser.add_argument("--output_dir", type=str, default="./outputs_vae_delta")
    parser.add_argument("--reasoning_frame", type=int, default=2)
    args = parser.parse_args()

    # Override for debugging (remove in production)
    args.image_path_ref = r"F:\ImageSet\ObjectRemoval\jimeng_RORD_qwen\portrait_output\1_F.png"
    args.image_path_target = r"F:\ImageSet\ObjectRemoval\jimeng_RORD_qwen\portrait_output\1_G.png"
    gamma = 1.25

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Enable faster compute on modern NVIDIA GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    pretrained_model_name_or_path = r"F:/T2ITrainer/qwen_models/qwen_image_edit_plus"
    vae = AutoencoderKLQwenImage.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    vae = vae.to(device, dtype=torch.float32)
    vae.eval()  # Important: disables unnecessary ops like dropout

    vae_scale_factor = 2 ** len(vae.temperal_downsample)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)
    latent_channels = 16

    # Load and move images to GPU
    image_ref = load_and_preprocess_image(image_processor, args.image_path_ref).to(device)
    image_tgt = load_and_preprocess_image(image_processor, args.image_path_target).to(device)

    # Encode images to latent space
    generator = torch.Generator(device=device).manual_seed(42)
    latent_ref = encode_vae_image(vae, latent_channels, image_ref, generator)
    latent_tgt = encode_vae_image(vae, latent_channels, image_tgt, generator)

    
    # Interpolate in latent space
    # interpolated_latents = linear_interpolation(latent_tgt, latent_ref, args.reasoning_frame, gamma=gamma)

    # # Decode and save each frame
    # for i, latent in enumerate(interpolated_latents):
    #     latent = latent.to(device)
    #     decoded_image = decode_vae_latent(vae, image_processor, latent)
    #     output_path = os.path.join(args.output_dir, f"interpolated_{i:03d}.png")
    #     decoded_image[0].save(output_path)
    #     print(f"Saved: {output_path}")

    # concat latent in width, decode and save one image
    # ori_interpolated_latents = [t.clone().detach() for t in interpolated_latents]
    # ori_interpolated_latents = torch.cat(ori_interpolated_latents, dim=4)
    # decoded_image = decode_vae_latent(vae, image_processor, ori_interpolated_latents)
    # output_path = os.path.join(args.output_dir, f"interpolated_concat.png")
    # decoded_image[0].save(output_path)
    # print(f"Saved: {output_path}")
    
    # interpolated_latents = interpolated_latents[::-1]  # Reverse: now goes from target → reference
    # interpolated_latents = torch.cat(interpolated_latents, dim=4)
    decoded_image = decode_vae_latent(vae, image_processor, latent_ref)
    output_path = os.path.join(args.output_dir, f"interpolated_concat.png")
    decoded_image[0].save(output_path)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()