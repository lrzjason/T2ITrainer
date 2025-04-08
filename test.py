import torch
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)

file = "F:/models/unet/fill_color_fix_beta.safetensors"
config = "F:/T2ITrainer/flux_models/fill/transformer/config.json"
transformer = FluxTransformer2DModel.from_single_file(file, config=config,  torch_dtype=torch.float16)

print("file loaded")