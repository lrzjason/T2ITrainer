
import torch
from diffusers import StableDiffusion3Pipeline
from hunyuandit.pipeline_hunyuandit import HunyuanDiTPipeline

from diffusers.utils import (
    convert_unet_state_dict_to_peft,
)

from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

import gc
import os
import copy

@torch.no_grad()
def main():
    pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers", torch_dtype=torch.float16)
    pipe.to("cuda")

    output_dir = f"F:/models/hy/"
    # lora_dir_name = "hy-open-30816"
    lora_dir_name = "hy-ori-prodigy"
    test_name = "amber"
    # prompt = "a young man draped in a dark hooded cloak, standing amidst bare trees with a mysterious expression on his face. He wears a pendant around his neck and has tousled hair. The setting appears to be a cold, overcast day in a forest or woodland area. The overall mood of the photo is somber and enigmatic."
    # prompt = "a close-up of what appears to be a dried orange, with intricate details revealing its translucent and textured surface. The colors range from deep reds to bright yellows, giving it a vibrant appearance against a light background. The patterns formed by the cells are reminiscent of stained glass windows."
    # prompt = "professional photo of a woman standing in front of a wall, posing for a picture. She is wearing no clothing, appearing naked in the scene. The wall behind her has a painting of a forest, adding an artistic touch to the setting. The woman's body is positioned in a way that she is leaning against the wall, creating a visually striking composition."
    # prompt = "highly detailed anime artwork, 1girl, ganyu_(genshin_impact), horns, solo, blue_hair, flower, looking_at_viewer, ahoge, qingxin_flower, holding_flower, bare_shoulders, holding, detached_sleeves, gloves, sidelocks, purple_eyes, blush, long_hair, white_flower, upper_body, black_gloves, "
    # prompt = "creative photo of Portrait of Rei Ayanami from Neon Genesis Evangelion, detailed scene, stunning details, anime, detailed environment, ray tracing, 8k"
    # prompt = "a young woman with blonde hair, smiling at the camera. She is wearing a light blue hoodie and jeans, standing on a sandy beach with footprints in the background. The sky appears clear, suggesting it might be late afternoon or early evening. The overall mood of the picture is cheerful and relaxed."
    # prompt = "Pikachu, a popular Pokémon character, dressed in an armored suit reminiscent of Iron Man's. He stands amidst a fiery and rocky environment with glowing embers surrounding him. The overall ambiance is intense and dramatic. The artwork appears to be digitally rendered, capturing intricate details and vibrant colors."
    # prompt = "sketch anime artwork, a side profile of a character with vibrant red hair, a black ribbon tied in a high ponytail, and a white shirt with a black vest. The character's eye is prominently red, and the art style appears to be a blend of realism and anime. The background is minimalistic, focusing mainly on the character' s profile. The composition is dynamic, with the character looking slightly upwards, and there's a sense of depth and perspective given to the character and the environment."
    # prompt = "sketch anime artwork, A female animated character ganyu with silver hair, purple eyes, and a brown outfit. She is depicted from a side angle, with her hair flowing gracefully, and there's a soft lighting effect, giving the image a dreamy ambiance. The background is white, emphasizing the character. The art style is detailed and intricate, with a blend of realism and fantasy elements. The artist's signature is present at the bottom right corner. The composition is balanced, with the character's face and hair occupying the central and right portions of the image."
    # prompt = "masterpiece anime artwork, a female animated character with long brown hair, wearing a red and white outfit with a ribbon on her head. She is posed against a background of stone steps, giving an impression of an outdoor setting. The character's pose is playful, with a hand gesture and a slight smile, suggesting a friendly and approachable demeanor. The composition focuses on the character, with the steps serving as a subtle background that adds depth to the image. The angle is slightly elevated, giving a view of the character from above, emphasizing her stature and the details of her attire."
    neg_prompt = "错误的眼睛，糟糕的人脸，毁容，糟糕的艺术，变形，多余的肢体，模糊的颜色，模糊，重复，病态，残缺，wrong eyes, bad faces, disfigurement, bad art, deformity, extra limbs, blurry colors, blur, repetition, sick, mutilation,"

    seed = torch.manual_seed(0)
    prompts_configs = [
        {
            "test_name":"c_woman_2",
            "prompt":"upperbody realistic portrait of a chinese woman with luscious, long black hair cascading down her back, gazing directly at the viewer with piercing black eyes that exude a sense of tranquility. Her right hand hold her chin, her left hand's index finger adorned with a delicate ring, as she reclines against a soft, white dress that seems to blend seamlessly into the blurry, dreamy background of lush greenery and vibrant flowers. Her ears sparkle with elegant earrings, adding a touch of sophistication to her serene and natural beauty. "
        },
        {
            "test_name":"poster_1",
            "prompt":"A photorealistic, close-up portrait of a woman with piercing brown eyes, thick black eyelashes, and a stunning ring adorning her finger, gazing directly at the viewer with an unflinching intensity. Her beautiful, realistic features are contrasted by a robotic or mechanical hand, covered in a red substance, that touches her cheek with an unsettling gentleness. The overall tone of the image is dark, intense, and dramatically captivating."
        },
        {
            "test_name":"poster_2",
            "prompt":"A monstrous creature with razor-sharp teeth, glistening with saliva, and a long, slithering tongue that darted in and out of its open mouth, revealing its razor-like teeth. It seems to be emerging from or hovering over a desolate, human-less cityscape during nighttime, with a dark and foreboding cloudy sky looming above, as if the monster had descended from the heavens themselves to terrorize the empty streets below."
        },
        # {
        #     "test_name":"c_woman",
        #     "prompt":"A solo, realistic Chinese woman with a serene and contemplative expression is captured in a stunning profile, her face a picture of elegance and refinement. Her long, luscious brown hair cascades down her back, framing her delicate features and accentuating her gentle, parted lips. Her brown eyes seem to hold a deep wisdom, gazing into the distance with a quiet introspection. The soft, warm lighting highlights the subtle details of her traditional attire, adorned with intricate designs that speak to her cultural heritage. Set against a blurry background, with distant lights blurred into a warm haze, the focus is drawn squarely to the woman's quiet, introspective beauty. "
        # },
        # {
        #     "test_name": "man",
        #     "prompt": "a young man draped in a dark hooded cloak, standing amidst bare trees with a mysterious expression on his face. He wears a pendant around his neck and has tousled hair. The setting appears to be a cold, overcast day in a forest or woodland area. The overall mood of the photo is somber and enigmatic."
        # },
        # {
        #     "test_name": "orange",
        #     "prompt": "a close-up of what appears to be a dried orange, with intricate details revealing its translucent and textured surface. The colors range from deep reds to bright yellows, giving it a vibrant appearance against a light background. The patterns formed by the cells are reminiscent of stained glass windows."
        # },
        # {
        #     "test_name": "nude",
        #     "prompt": "professional photo of a woman standing in front of a wall, posing for a picture. She is wearing no clothing, appearing naked in the scene. The wall behind her has a painting of a forest, adding an artistic touch to the setting. The woman's body is positioned in a way that she is leaning against the wall, creating a visually striking composition."
        # },
        # {
        #     "test_name": "ganyu",
        #     "prompt": "highly detailed anime artwork, 1girl, ganyu_(genshin_impact), horns, solo, blue_hair, flower, looking_at_viewer, ahoge, qingxin_flower, holding_flower, bare_shoulders, holding, detached_sleeves, gloves, sidelocks, purple_eyes, blush, long_hair, white_flower, upper_body, black_gloves, "
        # },
        # {
        #     "test_name": "rei",
        #     "prompt": "creative photo of Portrait of Rei Ayanami from Neon Genesis Evangelion, detailed scene, stunning details, anime, detailed environment, ray tracing, 8k"
        # },
        # {
        #     "test_name": "woman",
        #     "prompt": "a young woman with blonde hair, smiling at the camera. She is wearing a light blue hoodie and jeans, standing on a sandy beach with footprints in the background. The sky appears clear, suggesting it might be late afternoon or early evening. The overall mood of the picture is cheerful and relaxed."
        # },
        # {
        #     "test_name": "pika",
        #     "prompt": "Pikachu, a popular Pokémon character, dressed in an armored suit reminiscent of Iron Man's. He stands amidst a fiery and rocky environment with glowing embers surrounding him. The overall ambiance is intense and dramatic. The artwork appears to be digitally rendered, capturing intricate details and vibrant colors."
        # },
        # {
        #     "test_name": "diluc",
        #     "prompt": "sketch anime artwork, a side profile of a character with vibrant red hair, a black ribbon tied in a high ponytail, and a white shirt with a black vest. The character's eye is prominently red, and the art style appears to be a blend of realism and anime. The background is minimalistic, focusing mainly on the character' s profile. The composition is dynamic, with the character looking slightly upwards, and there's a sense of depth and perspective given to the character and the environment."
        # },
        # {
        #     "test_name": "ganyu-cog",
        #     "prompt": "sketch anime artwork, A female animated character ganyu with silver hair, purple eyes, and a brown outfit. She is depicted from a side angle, with her hair flowing gracefully, and there's a soft lighting effect, giving the image a dreamy ambiance. The background is white, emphasizing the character. The art style is detailed and intricate, with a blend of realism and fantasy elements. The artist's signature is present at the bottom right corner. The composition is balanced, with the character's face and hair occupying the central and right portions of the image."
        # },
    ]
    
    prompts_configs = [
        {
            "test_name":"zhaojinmai_zh3",
            "prompt":"赵今麦，一位亚洲女性，黑色长发披肩，戴着耳环，站在蓝色背景前。照片采用近景、平视和居中构图方式，呈现出真实摄影风格。照片中蕴含了人物摄影文化，同时展现了安静的氛围。"
        },
    ]
    # generation_configs = [
    #     {
    #         "lora_dir_name":"hy-ori",
    #     },
    #     {
    #         "lora_dir_name":"hy-lora-20544",
    #     },
    #     {
    #         "lora_dir_name":"hy-lora-30816",
    #     },
    #     {
    #         "lora_dir_name":"hy-lora-41088",
    #     },
    #     {
    #         "lora_dir_name":"hy-lora-51360",
    #     },
    #     {
    #         "lora_dir_name":"hy-lora-61632",
    #     },
    #     {
    #         "lora_dir_name":"hy-lora-71904",
    #     },
    #     {
    #         "lora_dir_name":"hy-lora-82176",
    #     },
    #     {
    #         "lora_dir_name":"hy-lora-92448",
    #     }
    # ]
    generation_configs = [
        {
            "lora_dir_name":"hy-zhaojinmai-600",
        },
        {
            "lora_dir_name":"hy-zhaojinmai-500",
        },
        {
            "lora_dir_name":"hy-zhaojinmai-400",
        },
        {
            "lora_dir_name":"hy-zhaojinmai-300",
        },
        {
            "lora_dir_name":"hy-zhaojinmai-200",
        },
        {
            "lora_dir_name":"hy-zhaojinmai-100",
        },
        
        # {
        #     "lora_dir_name":"hy-ori-prodigy",
        # },
        # {
        #     "lora_dir_name":"hy-openxl-4920",
        # },
        # {
        #     "lora_dir_name":"hy-openxl-3936",
        # },
        # {
        #     "lora_dir_name":"hy-openxl-3690",
        # },
        # {
        #     "lora_dir_name":"hy-openxl-2706",
        # },
        # {
        #     "lora_dir_name":"hy-openxl-1968",
        # },
        # {
        #     "lora_dir_name":"hy-openxl-2200",
        # },
        # {
        #     "lora_dir_name":"hy-openxl-1100",
        # },
    ]
    for generation_config in generation_configs:
        lora_dir_name = generation_config["lora_dir_name"]
        lora_path = f"{output_dir}{lora_dir_name}"
        lora_activate = False
        if os.path.exists(lora_path):
            transformer_ = pipe.transformer
            lora_path = f"{output_dir}{lora_dir_name}"
            lora_state_dict = HunyuanDiTPipeline.lora_state_dict(lora_path)
            HunyuanDiTPipeline.load_lora_into_transformer(lora_state_dict,transformer_)
            pipe.transformer = transformer_
            lora_activate = True
            
        for prompt_config in prompts_configs:
            test_name = prompt_config["test_name"]
            prompt = prompt_config["prompt"]
            image_path = f"img-{lora_dir_name}-{test_name}.png"
            if os.path.exists(image_path):
                continue
            # prompt = generation_config["prompt"]
            image = pipe(prompt,negative_prompt=neg_prompt,generator=seed).images[0]
            image.save(image_path)
            del image
            gc.collect()
            torch.cuda.empty_cache()
            
        if lora_activate:
            del lora_state_dict,transformer_
            pipe.unload_lora_weights()
        gc.collect()
        torch.cuda.empty_cache()
        

if __name__ == "__main__":
    main()