
import torch
from diffusers import HunyuanDiTPipeline, StableDiffusion3Pipeline

from diffusers.utils import (
    convert_unet_state_dict_to_peft,
)

from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers", torch_dtype=torch.float16)
pipe.to("cuda")

transformer_ = pipe.transformer

input_dir = "F:/models/hy/hy_test-800"
lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(input_dir)
HunyuanDiTPipeline.load_lora_into_transformer(lora_state_dict,transformer_)

pipe.transformer = transformer_
prompt = "cotton doll, A plush toy character with pink hair, a yellow crown adorned with a red gem, and large blue eyes. She is dressed in a white dress with a blue pendant around her neck. Surrounding her are cotton balls, which appear soft and fluffy. The backdrop is a solid peach color, providing contrast to the character's pastel hues. The overall composition gives off a warm and cute vibe, emphasizing the character's innocence and royalty."
# prompt = "cotton doll, A plush toy character of a beatiful woman"
image = pipe(prompt).images[0]

image.save("img.png")