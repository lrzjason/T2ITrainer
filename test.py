
import torch
from diffusers import StableDiffusion3Pipeline
from hunyuandit.pipeline_hunyuandit import HunyuanDiTPipeline

from diffusers.utils import (
    convert_unet_state_dict_to_peft,
)

from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers", torch_dtype=torch.float16)
pipe.to("cuda")

transformer_ = pipe.transformer

output_dir = f"F:/models/hy/"
lora_dir_name = "hy-dora-500"
prompt = "A stunning conceptual art piece showcasing a vibrant, fantastical magical sea creature. The creature, with a mix of both reptilian and aquatic features, has a mesmerizing iridescent blue and celestial purple color palette. It's adorned with elegant, flowing fabric, reminiscent of fashionable attire. The background is a magical, swirling combination of sky blue and celestial purple, creating a mesmerizing atmosphere. The image showcases a blend of fashion, fantasy, and art, evoking a sense of wonder and enchantment., conceptual art, photo, fashion, vibrant, painting, 3d render"


lora_path = f"{output_dir}{lora_dir_name}"
lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(lora_path)
HunyuanDiTPipeline.load_lora_into_transformer(lora_state_dict,transformer_)

pipe.transformer = transformer_
image = pipe(prompt).images[0]
image.save(f"img-{lora_dir_name}.png")