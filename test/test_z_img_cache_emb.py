import torch
import time
# from diffusers import ZImagePipeline
from z_img.ZImageLoraPipeline import ZImagePipeline
import gc
import os
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

from diffusers.models.transformers import ZImageTransformer2DModel

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
    model_path = r"F:\T2ITrainer_pulic\T2ITrainer\z_img_models\Z-Image-Turbo"
    prompt = """呈现出文艺的形象，年龄段在二十岁左右，脸型为柔和的类型。她有着黑色的长发，戴着一顶浅色的宽檐帽，身着蓝色的长裙，肩上挎着一个棕色的包。体态优雅，姿势是在木质的廊架下漫步，头部微微抬起，目光看向廊架上方，双手轻轻提起裙摆两侧。镜头采用平视角度，以中等距离拍摄，前景有绿色的枝叶和橙色的花朵作为虚化的点缀，清晰展现出她在廊架（周围有草地、树木等自然景色）中的姿态，整体氛围清新且富有诗意，画面中有文字'几番起伏 总不平'。"""
    base_embedding_path = "zimage_base_embedding.pt"
    
    seed = 43

    timings = {}  # Store timings for each stage

    # ============ Stage 1: Encode Prompt (or load cached) ============
    start_time = time.time()
    if os.path.exists(base_embedding_path):
        print("Loading cached prompt embeddings...")
        base_embedding = torch.load(base_embedding_path)
        prompt_embeds = base_embedding["prompt_embeds"]
    else:
        print("Encoding prompt...")
        text_pipeline = ZImagePipeline.from_pretrained(
            model_path,
            transformer=None,
            vae=None,
            torch_dtype=dtype,
        ).to(device)
        
        with torch.no_grad():
            prompt_embeds, _ = text_pipeline.encode_prompt(prompt)
        
        base_embedding = {
            "prompt_embeds": prompt_embeds,
        }
        torch.save(base_embedding, base_embedding_path)
        del text_pipeline
        flush()
    timings["Prompt Encoding"] = time.time() - start_time

    # ============ Stage 2: Load Full Pipeline ============
    start_time = time.time()
    print("Loading ZImage pipeline...")
    pipe = ZImagePipeline.from_pretrained(
        model_path,
        # transformer=None,
        text_encoder=None,
        tokenizer=None,
        torch_dtype=dtype,
    ).to(device)
    
    # transformer_path = r"F:\models\unet\Z-Image-Turbo-Merged.safetensors"
    # # load state dict from safetensors
    # pipe.transformer = ZImageTransformer2DModel.from_single_file(transformer_path).to(device)
    
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
    step = 8
    pipe = pipe.to(device, dtype)
    generator = torch.Generator(device=device).manual_seed(seed)
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

    loras = [
        # ("Original", None, step),
        # ("z_timesteps_adapter-39-30000", ["z_timesteps_adapter-39-30000"], step),
        # ("z_no_adapter-28-5017", ["z_no_adapter-28-5017"], step),
        # ("z_timesteps_adapter-7-5400", ["z_timesteps_adapter-7-5400"], step),
        # ("z_timesteps_adapter-13-10500", ["z_timesteps_adapter-13-10500"], step),
        # ("z_timesteps_adapter-7-6000", ["z_timesteps_adapter-7-6000"], step),
        ("z_timesteps_adapter-3-3000", ["z_timesteps_adapter-3-3000"], step),
        
        
        
        # ("z_timesteps_adapter-9-6750_converted", ["z_timesteps_adapter-9-6750_converted"], step),
        # ("mixed", ["z_timesteps_adapter-9-6750_converted", "z_with_transformer_adapter-28-5017"], step)
        # ("zp_with_kl_hr-2-1035", 9),
        # ("zp_with_kl_hr-2-1035", 15),
        # ("zp_with_kl_hr-2-1035", 18),
        # ("zp_with_kl_hr-2-1035", 21)
        # "z_photograph-5-1038",
        # "z_photograph_disabled_last8",
        # "z_photograph_disabled_last10",
        # "z_photograph_disabled_first10"
        # "z_photograph_with_kl-9-1730"
    ]
    
    def generate_image_with_lora(idx, weight_name, lora_list, steps):
        lora_path = r"F:\models\Lora\z_image"
        output_path = f"z_{idx}_{weight_name}_{steps}.png"
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
            pipe.fuse_lora(adapter_names=lora_list, lora_scale=1)

        start_time = time.time()
        print("Generating image with LoRA...")
        # steps = 9
        image = pipe(
            prompt_embeds=prompt_embeds,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]
        generation_time = time.time() - start_time

        pipe.unload_lora_weights()
        # Save result
        image.save(output_path)
        return generation_time
    
    for idx, (lora_name, lora_list, steps) in enumerate(loras):
        timings[f"LoRA {lora_name}"] = generate_image_with_lora(idx+1, lora_name, lora_list, steps)
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