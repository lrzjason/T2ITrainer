from qwen.transformer_qwenimage import BlockSwapQwenImageTransformer2DModel
import torch
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImagePipeline
)

model_path = r"F:\models\unet\Rebalance_v1.safetensors"

pretrain_dir = r"F:\T2ITrainer_pulic\T2ITrainer\qwen_models\qwen_image_nf4"
output_dir = r"F:\T2ITrainer_pulic\T2ITrainer\qwen_models\rebalance_v1"

weight_dtype = torch.bfloat16
offload_device = torch.device("cuda")

pretrain_subdir = r"F:\T2ITrainer_pulic\T2ITrainer\qwen_models\qwen_image_nf4\transformer"
config = BlockSwapQwenImageTransformer2DModel.load_config(pretrain_subdir)

transformer = BlockSwapQwenImageTransformer2DModel.from_single_file(model_path, 
                    config=config,  
                    torch_dtype=weight_dtype
                ).to(offload_device)

pipeline = QwenImagePipeline.from_pretrained(
    pretrain_dir,
    vae=None,
    transformer=None,
    text_encoder=None,
    tokenizer=None,
    torch_dtype=torch.bfloat16,
).to(offload_device)

pipeline.transformer = transformer

pipeline.save_pretrained(output_dir)