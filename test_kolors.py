
import torch
from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline

from diffusers.utils import (
    convert_unet_state_dict_to_peft,
)

from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers import EulerDiscreteScheduler,DDPMScheduler

root_dir = "F:/Kolors_sample/"
ckpt_dir = f'{root_dir}weights/Kolors'
text_encoder = ChatGLMModel.from_pretrained(
    f'{ckpt_dir}/text_encoder',
    torch_dtype=torch.float16).half()
tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).half()
# scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
scheduler = DDPMScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).half()
pipe = StableDiffusionXLPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            force_zeros_for_empty_prompt=False)
pipe.to("cuda")
pipe.enable_model_cpu_offload()

# transformer_ = pipe.transformer

output_dir = f"F:/models/kolors/"
lora_dir_name = "kolors_test-4000"
# prompt = "A captivating futuristic portrait of a young woman with half of her face illuminated in bright blue and the other half in vibrant yellow. The contrasting colors create a striking visual effect, accentuating her unique appearance. Perched on her head is a glowing wreath of light bulbs, casting a warm and inviting light. The city skyline is visible in the background, barely lit, giving the impression of a serene and tranquil atmosphere. The overall composition exudes a sense of peace and inspiration, inviting viewers to explore the depths of this futuristic world., portrait photography, photo"
# prompt = "洛夫克拉夫特式的利维坦恐怖袭击小镇"
# prompt = "cotton doll, 洛夫克拉夫特式的利维坦恐怖袭击小镇 lovecraftian leviathan horror lumbering over a town"
prompt = "赵今麦，一位亚洲女性，黑色长发披肩，戴着耳环，站在蓝色背景前。照片采用近景、平视和居中构图方式，呈现出真实摄影风格。照片中蕴含了人物摄影文化，同时展现了安静的氛围。"
neg_prompt = "错误的眼睛，糟糕的人脸，毁容，糟糕的艺术，变形，多余的肢体，模糊的颜色，模糊，重复，病态，残缺，wrong eyes, bad faces, disfigurement, bad art, deformity, extra limbs, blurry colors, blur, repetition, sick, mutilation,"

seed = torch.manual_seed(0)

# lora_path = f"{output_dir}{lora_dir_name}"
# pipe.load_lora_weights(lora_path, adapter_name="test")
# pipe.fuse_lora(lora_scale=0.7)
# lora_state_dict = StableDiffusionXLPipeline.lora_state_dict(lora_path)
# StableDiffusionXLPipeline.load_lora_into_transformer(lora_state_dict,transformer_)

# pipe.transformer = transformer_
image = pipe(prompt,negative_prompt=neg_prompt,generator=seed).images[0]
# image = pipe(prompt).images[0]
image.save(f"img-{lora_dir_name}.png")