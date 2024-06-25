
import torch
from diffusers import HunyuanDiTPipeline, StableDiffusion3Pipeline

from diffusers.utils import (
    convert_unet_state_dict_to_peft,
)

from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers", torch_dtype=torch.float16)
pipe.to("cuda")

transformer_ = pipe.transformer

output_dir = f"F:/models/hy/"
lora_dir_name = "hy-dora-600"
prompt = "cotton doll, A plush toy character of a blonde policy officer"


lora_path = f"{output_dir}{lora_dir_name}"
lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(lora_path)
HunyuanDiTPipeline.load_lora_into_transformer(lora_state_dict,transformer_)

pipe.transformer = transformer_
image = pipe(prompt).images[0]
image.save(f"img-{lora_dir_name}.png")