import torch
import time
# from diffusers import ZImagePipeline
from longcat.pipeline_longcat_image import LongCatImagePipeline
from longcat.pipeline_longcat_image_edit import LongCatImageEditPipeline
import gc
import os
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import AutoProcessor

from longcat.longcat_image_dit import LongCatImageTransformer2DModel

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

@torch.no_grad()
def main():
    
    # quant_config = TransformersBitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    # )
    
    width = 1024
    height = 1536

    device = torch.device("cuda")
    dtype = torch.bfloat16
    model_path = r"F:\T2ITrainer_pulic\T2ITrainer\longcat_models\LongCat-Image"
    prompt = """采用俯视角度拍摄，为中景画面。画面呈现出一位女性，她有着黑色的长发，头发柔顺，背后有白色的天使翅膀，翅膀羽毛纹理清晰，她穿着浅色的连衣裙，姿态是坐在高处的栏杆上，俯瞰下方的城市街道。下方的城市街道充满了繁华的景象，有众多的人群、行驶的车辆（包括黄色的出租车），还有林立的高楼与明亮的广告牌，营造出一种热闹的都市氛围。画面中没有风，不存在慢门效果，也没有动态模糊，整体画面清晰，将女性的长发、天使翅膀的细节、连衣裙的颜色，以及城市街道的人群、车辆、建筑与广告牌等细节都很好地展现出来，从头发的黑色光泽、翅膀的白色羽毛质感、连衣裙的浅色调，到人群的密集、车辆的色彩与形态、建筑的高耸以及广告牌的明亮与多样，都清晰可见，营造出一种神秘、梦幻且与繁华都市形成对比的画面效果。"""
    prompt_list = [
        "采用中距离镜头，略微仰视的角度拍摄。主体是一位有着白色长发（发尾带有粉色）的机械女性，她身着以深蓝色为主、带有金色和粉色细节的机械装甲，装甲上有多处机械关节和结构，腿部装甲上有白色的“77”标识及其他文字。她呈现出一种俯身的动态姿态，双手和双脚（其中一只脚为高跟机械结构）支撑在地面，头发飘逸，粉色的眼睛注视着前方。",
        "艺术摄影，强烈明暗对比，A captivating black and white close-up portrait of a beautiful young Asian woman with flawless skin, short bob haircut featuring straight blunt bangs framing her forehead, eyes gently closed with long lashes, subtle pouty lips, serene and introspective expression, wearing a form-fitting deep V-neck black ribbed sweater that accentuates her, a delicate thin necklace with a small pendant, dramatic side lighting from the left creating strong shadows and highlights on her face, neck, and shoulders, high contrast chiaroscuro style, minimalist studio background with soft gradient from light to dark, photorealistic, vintage film grain effect, high detail, emotional and mysterious atmosphere.",
        "动态感，灵动。惊艳，艳丽，五官精致，裙摆飞舞动态模糊，大雪模糊，少女，皮肤白皙有光泽。长发及腰，积雪，毛边白色汉服，亮片浅红花朵点缀在衣服上，大雪纷飞。头发光泽度。古风。绝美少女。大眼睛，大师级构图，黑发，色彩对比度，空间感，发丝飞扬，角度，视角，柔焦，柔和，唯美，梦幻，朦胧美。符合场景的动作，对比度，色彩饱满的。冷色调。聚焦朦胧，雪花，浅蓝色刺绣油纸伞，",
        "真人写真， 20岁女生，故事感 俯视镜头视角，女孩，面容精致，面部特写，黑茶色长发，发型分层剪裁，外层及耳短发呈蓬松弧状包裹内层，发尾保留整体长发垂坠感，内外层通过层次落差形成立体轮廓，侧脸，露肩低领衣服，肌肤雪白有光泽。电影质感，超清画质，超现实主义写真摄影，运用伦勃朗光",
        "采用特写镜头，近距离聚焦主体。主体是一位有着黑色短发的女性，她穿着白色的上衣，耳部佩戴着金色的耳环，耳环上有复杂的装饰细节。画面中有一道彩虹色的光影，从她的面部延伸到上衣部分，光影呈现出多种色彩的渐变效果。背景是浅灰色的纯色背景，使得主体和彩虹光影更加突出。她的头发纹理、上衣的质感、耳环的细节以及彩虹光影的色彩层次都清晰可见，整体画面通过特写镜头将主体的形象、服饰细节以及光影效果都展现得十分细致，每一处元素，如头发的发丝走向、上衣的褶皱、耳环的装饰纹理以及彩虹光影的色彩分布等都能被清楚观察到。",
        "深邃的黑暗中，墨色如血般泼洒晕染，糅合了洒蓝与褐色的混沌基调，勾勒出一个成熟男性极具视觉冲击力的半张脸。金粉与朱砂粒子，高景深构图强化了戏剧张力，微距之下每一寸肌肤纹理皆覆有细腻的磨砂质感，在完美光影的雕琢下尽显超高清的细节刻画。金箔浮光暗涌，朱砂点缀如血，整体弥漫着深邃的暗黑美学气息，堪称一件充满张力的大师级作品。",
        "后背视角，一个穿着中式传统风格连衣裙的年轻女子在跳舞的超现实照片，她穿着一个半透明轻薄的，蓝色和白色分层相间的多层次的中式传统风格元素连衣裙，裙子上有蓝色和白色花卉的印染图案，肩部和背部裸露。她拿着一个蓝色纸扇在她的右手，在她的头上举起。她的长，深棕色的头发用蓝色的花装饰，头发和半透明裙子的裙摆被风吹起在空中飘动。她的转向右边，闭眼，侧面对镜头。蓝色的花瓣散布在她周围，一只蓝色的鸟在她的头附近飞行。(背景是翻腾的白色和蓝色海浪)，(裙摆与脚下溅起的大量水花融为一体)，略微模糊，聚焦在女人全身姿态。灯光明亮均匀，没有刺眼的阴影。相机角度略低，偏向侧面。景深较浅，背景柔和模糊。图像使用自然灯光效果来增强细腻和通风的感觉。艺术照，胶片摄影风格, 翩若惊鸿，婉若游龙。荣曜秋菊，华茂春松。髣髴兮若轻云之蔽月，飘飖兮若流风之回雪。远而望之，皎若太阳升朝霞；迫而察之，灼若芙蕖出渌波。秾纤得中，修短合度。肩若削成，腰如约素。延颈秀项，皓质呈露。芳泽无加，铅华弗御。云髻峨峨，修眉联娟。丹唇外朗，皓齿内鲜。明眸善睐，靥辅承权。瓌姿艳逸，仪静体闲。柔情绰态，媚于语言。",
        "写实照片，摄影风格，光圈，暗角，她雪白修长的颈部被深蓝丝绒choker死死勒出一圈诱人红痕，中央那颗硕大黄宝石吊坠沉甸甸地坠在锁骨下方，随着急促的呼吸上下晃动，随时可能撞击到已经挺立发硬的粉嫩乳头。两侧细长绑带绕到颈后打成蝴蝶结，仿佛只要轻轻一扯就能让她彻底窒息般臣服。上身那点可怜的“布料”早已被彻底扯开，深蓝绸缎挂脖绳只剩一根细线勒在脖颈，两团雪白饱满的G杯巨乳完全裸露在外，乳晕是最娇艳的樱桃粉。她右手正毫不遮掩地揉捏自己左乳，指尖掐住乳头狠狠拉扯，乳肉在指缝间溢出淫靡的变形；左手则向下探去，两指已深深插入早已泛滥成灾的粉嫩蜜穴，带出晶莹的淫液拉丝。",
        "剪影，隐约的淡彩+红色，逆朦，泛朦，对焦模糊，一个灵动纯美的EVA绫波丽模糊人像及暗红色场景，浅色短发，突出动态发丝的凌乱美，超高颜值，伦勃朗光，反射，折射，肌理质感，泛光模糊晕染，高噪点，胶片颗粒质感，极简风，柔焦美学，情绪写意，前卫视觉艺术美学，大片高级感，电影美学，大师杰作",
        "人物插画，叠涂，剑芒，弥散渐变，色彩饱满有层次。古风 模糊深色背景，画面昏暗，阴冷。白衣少女，手指缠着发光的红线，诡异民俗风格，红线，小铜钱，流动感，神秘，氛围感拉满，模糊雾气，冷色调。繁复垂坠，画面光影明暗对比。模糊红线前景。倾斜构图，仰视角度。",
        "暗黑调色，阴暗布景，皮肤粗糙，采用特写镜头，近距离聚焦主体。主体是一位身着深色西装（含马甲）、佩戴带有花纹的蓝色领带的男性，他手持雪茄，雪茄烟雾呈白色缭绕，佩戴着眼镜，坐在黑色的皮质座椅上。背景是深色的室内环境，带有一些模糊的装饰元素，前景的桌面上有文件和金色的摆件。他的西装细节、领带的花纹、雪茄的纹理以及烟雾的形态都清晰可见，整体通过特写镜头将主体的姿态、服饰细节以及周围的烟雾、室内元素都展现得十分细致，每一处元素，如西装的剪裁、领带的图案、雪茄的色泽、烟雾的飘动轨迹、座椅的质感以及桌面物品的轮廓等都能被清楚观察到。"
    ]
    embedding_list = [
        
    ]
    seed = 42

    timings = {}  # Store timings for each stage

    text_pipeline = None
    for idx, prompt in enumerate(prompt_list):
        base_embedding_path = f"z_longcat_embedding_{idx}.pt"
        if not os.path.exists(base_embedding_path):
            text_processor = AutoProcessor.from_pretrained( model_path, subfolder = 'tokenizer'  )
            print("Encoding prompt...")
            text_pipeline = LongCatImagePipeline.from_pretrained(
                model_path,
                transformer=None,
                text_processor=text_processor,
                vae=None,
                torch_dtype=dtype,
            ).to(device)
            break
    
    # base_embedding_path = "z_longcat_embedding.pt"
    for idx, prompt in enumerate(prompt_list):
        base_embedding_path = f"z_longcat_embedding_{idx}.pt"
        # base_embedding_path_list.append(base_embedding_path)
    
        # ============ Stage 1: Encode Prompt (or load cached) ============
        start_time = time.time()
        if os.path.exists(base_embedding_path):
            print("Loading cached prompt embeddings...")
            base_embedding = torch.load(base_embedding_path)
            # prompt_embeds = base_embedding["prompt_embeds"]
            # text_ids = base_embedding["text_ids"]
            embedding_list.append(base_embedding)
        else:
            # check if rewritten_prompt exists
            if os.path.exists(f"rewritten_prompt_{idx}.txt"):
                with open(f"rewritten_prompt_{idx}.txt", "r", encoding="utf-8") as f:
                    prompt = f.read()
            else:
                print("Rewriting prompt...")
                # rewrite prompt
                prompt = text_pipeline.rewire_prompt(prompt, device )
                
                # save rewritten prompt as text file
                with open(f"z_rewritten_prompt_{idx}.txt", "w", encoding="utf-8") as f:
                    f.write(prompt)
            
            with torch.no_grad():
                prompt_embeds,  text_ids = text_pipeline.encode_prompt([prompt], device=device, dtype=dtype)
            
            base_embedding = {
                "prompt_embeds": prompt_embeds,
                "text_ids": text_ids
            }
            embedding_list.append(base_embedding)
            torch.save(base_embedding, base_embedding_path)
            del base_embedding
            flush()
    timings["Prompt Encoding"] = time.time() - start_time
    if text_pipeline is not None:  # Free up memory
        del text_pipeline

    # ============ Stage 2: Load Full Pipeline ============
    start_time = time.time()
    print("Loading ZImage pipeline...")
    pipe = LongCatImagePipeline.from_pretrained(
        model_path,
        transformer=None,
        text_encoder=None,
        tokenizer=None,
        torch_dtype=dtype,
    ).to(device)
    
    # load state dict from safetensors
    transformer_folder = os.path.join(model_path, "transformer")
    pipe.transformer = LongCatImageTransformer2DModel.from_pretrained(
        transformer_folder, 
        ).to(device)
    
    pipe.enable_model_cpu_offload()

    # Enable optimizations — uncomment quantize if needed later
    # print("Optimizing transformer...")
    # quantize(pipe.transformer, weights=qint4)
    # freeze(pipe.transformer)
    # pipe.enable_model_cpu_offload()
    timings["Pipeline Loading"] = time.time() - start_time

    # ============ Stage 3: Image Generation ============
    start_time = time.time()
    print("Generating image...")
    step = 30
    pipe = pipe.to(device, dtype)
    # image = pipe(
    #     prompt_embeds=prompt_embeds,
    #     height=height,
    #     width=width,
    #     num_inference_steps=step,
    #     guidance_scale=0.0,
    #     generator=generator,
    # ).images[0]
    # timings["Image Generation"] = time.time() - start_time
    # output_path = f"1_zimage_{step}steps.png"
    # # Save result
    # image.save(output_path)
    # print(f"Image saved to {output_path}")
    
    lora_path = r"F:\models\Lora\longcat"
    loras = [
        ("Original", None, step),
        # ("zturbo_adapter-0-2700", ["zturbo_adapter-0-2700"], step),
        # ("zturbo_adapter-3-10800", ["zturbo_adapter-3-10800"], step),
        ("zturbo_adapter-4-13500", ["zturbo_adapter-4-13500"], step),
        ("zturbo_adapter-5-16200", ["zturbo_adapter-5-16200"], step),
        ("zturbo_adapter-6-18900", ["zturbo_adapter-6-18900"], step),
        ("zturbo_adapter-7-21600", ["zturbo_adapter-7-21600"], step),
        ("zturbo_adapter-9-27000", ["zturbo_adapter-9-27000"], step),
        
    ]
    
    def generate_image_with_lora(e_idx, idx, prompt_embeds, text_ids, weight_name, lora_list, steps):
        # ensure same seed
        generator = torch.Generator(device=device).manual_seed(seed)
        output_path = f"z_{e_idx}_{idx}_{weight_name}_{steps}.png"
        if os.path.exists(output_path):
            print(f"Image already exists: {output_path}")
            return 0
        if weight_name != "Original":
            for lora_name in lora_list:
                pipe.load_lora_weights(
                    lora_path,
                    weight_name=f"{lora_name}.safetensors",
                    adapter_name=lora_name
                )
                print("load_lora", lora_name)
            # pipe.fuse_lora(adapter_names=lora_list, lora_scale=1)
            pipe.set_adapters(lora_list)

        start_time = time.time()
        print("Generating image with LoRA...")
        # steps = 9
        image = pipe(
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            height=height,
            width=width,
            num_inference_steps=steps,
            generator=generator,
            guidance_scale=2.5,
            num_images_per_prompt=1,
            enable_cfg_renorm=True,
            enable_prompt_rewrite=False  # Reusing the text encoder as a built-in prompt rewriter
        ).images[0]
        generation_time = time.time() - start_time

        pipe.unload_lora_weights()
        # Save result
        image.save(output_path)
        return generation_time
    
    for e_idx, embedding in enumerate(embedding_list):
        prompt_embeds = embedding["prompt_embeds"]
        text_ids = embedding["text_ids"]
        for idx, (lora_name, lora_list, steps) in enumerate(loras):
            timings[f"LoRA {lora_name}"] = generate_image_with_lora(e_idx, idx, prompt_embeds, text_ids, lora_name, lora_list, steps)
        del prompt_embeds, text_ids
    # ============ Final Cleanup ============
    del pipe
    flush()

    # ============ Print Timing Summary ============
    print("\n=== Timing Summary ===")
    total_time = 0.0
    for stage, t in timings.items():
        print(f"{stage:20s}: {t:.2f} seconds")
        total_time += t
    print(f"{'Total':20s}: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()