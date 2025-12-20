import os
import argparse
import torch
import torchvision.transforms as T
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKLQwenImage

pretrained_model_name_or_path = r"F:/T2ITrainer/qwen_models/qwen_image_edit_plus"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    pretrained_model_name_or_path, subfolder="scheduler"
)

vae = AutoencoderKLQwenImage.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="vae",
).to(device)
vae.eval()

# handle guidance
def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

def compute_delta_feature_latent(
    target_latent: torch.Tensor,   # [B, C, H, W] in latent space
    noise_latent: torch.Tensor,    # same shape
    t: float,
    t_steps: list,
    lambda_decay: float = 0.6
):
    B, C, H, W = target_latent.shape

    t_int = torch.tensor([round(t)], device=device, dtype=torch.long)
    t_steps_int = torch.tensor([round(tn) for tn in t_steps], device=device, dtype=torch.long)

    t_int = torch.clamp(t_int, 1, 1000)
    t_steps_int = torch.clamp(t_steps_int, 1, 1000)

    sigma_t = get_sigmas(t_int, n_dim=4, dtype=target_latent.dtype)      # [1, 1, 1, 1]
    sigma_tn_list = get_sigmas(t_steps_int, n_dim=4, dtype=target_latent.dtype)  # [N, 1, 1, 1]

    sigma_t = sigma_t.expand(B, C, H, W)
    x_t = (1.0 - sigma_t) * target_latent + sigma_t * noise_latent

    delta_sum = torch.zeros_like(x_t)
    total_weight = 0.0

    for n in range(len(t_steps)):
        sigma_tn = sigma_tn_list[n].expand(B, C, H, W)
        x_tn = (1.0 - sigma_tn) * target_latent + sigma_tn * noise_latent
        diff = x_t - x_tn
        weight = (lambda_decay ** n)
        delta_sum += weight * diff
        total_weight += weight

    delta = delta_sum / total_weight
    return delta, x_t, x_tn

def load_image_as_tensor(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert('RGB')
    transform = T.ToTensor()
    return transform(img).unsqueeze(0)  # [1, 3, H, W]

def decode_latent_to_image(latent: torch.Tensor, vae) -> torch.Tensor:
    with torch.no_grad():
        image = vae.decode(latent).sample
    image = (image / 2.0 + 0.5).clamp(0, 1)  # [-1,1] → [0,1]
    return image

def save_rgb_image(tensor: torch.Tensor, save_path: str):
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.shape[0] != 3:
        raise ValueError(f"Expected 3-channel RGB tensor, got {tensor.shape}")
    img_array = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    Image.fromarray(img_array, mode='RGB').save(save_path)
    print(f"Saved RGB image to {save_path}")

def save_delta_as_image(delta: torch.Tensor, save_path: str):
    if delta.dim() == 4:
        delta = delta.squeeze(0)
    delta_mag = delta.abs().mean(dim=0)
    delta_norm = (delta_mag - delta_mag.min()) / (delta_mag.max() - delta_mag.min() + 1e-8)
    img_array = (delta_norm * 255).clamp(0, 255).byte().cpu().numpy()
    Image.fromarray(img_array, mode='L').save(save_path)
    print(f"Saved delta magnitude to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, help="Path to clean target image")
    parser.add_argument("--output_dir", type=str, default="./outputs_vae_delta", help="Directory to save results")
    parser.add_argument("--t", type=float, default=0.3, help="Current time step in [0,1]")
    parser.add_argument("--t_steps", type=float, nargs='+', default=[0.4, 0.5, 0.6], help="Future time steps in [0,1]")
    parser.add_argument("--lambda_decay", type=float, default=0.6, help="Temporal decay factor")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    args.image_path = r"F:/ImageSet/ObjectRemoval_test/drink.jpg"

    # Load and preprocess image
    image_tensor = load_image_as_tensor(args.image_path).to(device)  # [1, 3, H, W]

    # Encode to latent
    with torch.no_grad():
        target_latent = vae.encode(image_tensor).latent_dist.sample()  # [1, C, H', W'], C=4 typically

    # Generate noise in latent space
    noise_latent = torch.randn_like(target_latent)

    # Scale timesteps to [1, 1000]
    scaled_t = args.t * 1000
    scaled_t_steps = [t_val * 1000 for t_val in args.t_steps]

    if not all(tn > scaled_t for tn in scaled_t_steps):
        raise ValueError(f"All t_steps must be > t. Got t={scaled_t}, t_steps={scaled_t_steps}")

    # Compute delta in latent space
    delta_latent, x_t_latent, x_tn_latent = compute_delta_feature_latent(
        target_latent=target_latent,
        noise_latent=noise_latent,
        t=scaled_t,
        t_steps=scaled_t_steps,
        lambda_decay=args.lambda_decay
    )

    # Decode back to image space
    delta_img = decode_latent_to_image(delta_latent, vae)
    x_t_img = decode_latent_to_image(x_t_latent, vae)
    x_tn_img = decode_latent_to_image(x_tn_latent, vae)

    # Save results
    save_rgb_image(x_t_img, os.path.join(args.output_dir, "x_t_decoded.png"))
    save_rgb_image(x_tn_img, os.path.join(args.output_dir, "x_tn_decoded.png"))
    save_delta_as_image(delta_latent, os.path.join(args.output_dir, "delta_latent_mag.png"))

    # Optional: save raw delta latent as RGB (for debugging latent channels)
    if delta_latent.shape[1] == 3:
        save_rgb_image(decode_latent_to_image(delta_latent, vae), os.path.join(args.output_dir, "delta_decoded.png"))
    else:
        print("Delta latent has non-RGB channels; skipping decoded delta RGB save.")

    print("✅ Done: Delta computed in VAE latent space and decoded back to image!")

if __name__ == "__main__":
    main()