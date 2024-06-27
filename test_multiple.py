
import torch
from diffusers import StableDiffusion3Pipeline
from hunyuandit.pipeline_hunyuandit import HunyuanDiTPipeline

from diffusers.utils import (
    convert_unet_state_dict_to_peft,
)

from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

import gc

pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers", torch_dtype=torch.float16)
pipe.to("cuda")

transformer_ = pipe.transformer

output_dir = f"F:/models/hy/"
# lora_dir_name = "hy-open-30816"
lora_dir_name = "hy-ori"
# prompt = "A captivating futuristic portrait of a young woman with half of her face illuminated in bright blue and the other half in vibrant yellow. The contrasting colors create a striking visual effect, accentuating her unique appearance. Perched on her head is a glowing wreath of light bulbs, casting a warm and inviting light. The city skyline is visible in the background, barely lit, giving the impression of a serene and tranquil atmosphere. The overall composition exudes a sense of peace and inspiration, inviting viewers to explore the depths of this futuristic world., portrait photography, photo"
# prompt = "洛夫克拉夫特式的利维坦恐怖袭击小镇"
# prompt = "cotton doll, 洛夫克拉夫特式的利维坦恐怖袭击小镇 lovecraftian leviathan horror lumbering over a town"
# prompt = "cotton doll, A plush toy character with pink hair, a yellow crown adorned with a red gem, and large blue eyes. She is dressed in a white dress with a blue pendant around her neck. Surrounding her are cotton balls, which appear soft and fluffy. The backdrop is a solid peach color, providing contrast to the character's pastel hues. The overall composition gives off a warm and cute vibe, emphasizing the character's innocence and royalty."
prompt = "highly detailed anime artwork, 1girl, ganyu_(genshin_impact), horns, solo, blue_hair, flower, looking_at_viewer, ahoge, qingxin_flower, holding_flower, bare_shoulders, holding, detached_sleeves, gloves, sidelocks, purple_eyes, blush, long_hair, white_flower, upper_body, black_gloves, "
neg_prompt = "错误的眼睛，糟糕的人脸，毁容，糟糕的艺术，变形，多余的肢体，模糊的颜色，模糊，重复，病态，残缺，wrong eyes, bad faces, disfigurement, bad art, deformity, extra limbs, blurry colors, blur, repetition, sick, mutilation,"

seed = torch.manual_seed(0)

image = pipe(prompt,negative_prompt=neg_prompt,generator=seed).images[0]
image.save(f"img-{lora_dir_name}.png")

lora_dir_name = "hy-open-20544"

lora_path = f"{output_dir}{lora_dir_name}"
lora_state_dict = HunyuanDiTPipeline.lora_state_dict(lora_path)
HunyuanDiTPipeline.load_lora_into_transformer(lora_state_dict,transformer_)
pipe.transformer = transformer_

image = pipe(prompt,negative_prompt=neg_prompt,generator=seed).images[0]
image.save(f"img-{lora_dir_name}.png")

del lora_state_dict,transformer_
gc.collect()
torch.cuda.empty_cache()

transformer_ = pipe.transformer
lora_dir_name = "hy-open-30816"

lora_path = f"{output_dir}{lora_dir_name}"
lora_state_dict = HunyuanDiTPipeline.lora_state_dict(lora_path)
HunyuanDiTPipeline.load_lora_into_transformer(lora_state_dict,transformer_)
pipe.transformer = transformer_

image = pipe(prompt,negative_prompt=neg_prompt,generator=seed).images[0]
image.save(f"img-{lora_dir_name}.png")

del lora_state_dict,transformer_
gc.collect()
torch.cuda.empty_cache()

transformer_ = pipe.transformer
lora_dir_name = "hy-open-41088"

lora_path = f"{output_dir}{lora_dir_name}"
lora_state_dict = HunyuanDiTPipeline.lora_state_dict(lora_path)
HunyuanDiTPipeline.load_lora_into_transformer(lora_state_dict,transformer_)
pipe.transformer = transformer_

image = pipe(prompt,negative_prompt=neg_prompt,generator=seed).images[0]
image.save(f"img-{lora_dir_name}.png")

del lora_state_dict,transformer_
gc.collect()
torch.cuda.empty_cache()

transformer_ = pipe.transformer
lora_dir_name = "hy-open-51360"

lora_path = f"{output_dir}{lora_dir_name}"
lora_state_dict = HunyuanDiTPipeline.lora_state_dict(lora_path)
HunyuanDiTPipeline.load_lora_into_transformer(lora_state_dict,transformer_)
pipe.transformer = transformer_

image = pipe(prompt,negative_prompt=neg_prompt,generator=seed).images[0]
image.save(f"img-{lora_dir_name}.png")

