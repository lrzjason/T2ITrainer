import torch
import time
# from diffusers import ZImagePipeline
# from longcat.pipeline_longcat_image import LongCatImagePipeline
from longcat.pipeline_longcat_image_edit import LongCatImageEditPipeline
import gc
import os
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import AutoProcessor

from longcat.longcat_image_dit import LongCatImageTransformer2DModel
from utils.image_utils_longcat import crop_image

from PIL import Image

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
    
    # width = 1024
    # height = 1536

    device = torch.device("cuda")
    dtype = torch.bfloat16
    model_path = r"F:\T2ITrainer_pulic\T2ITrainer\longcat_models\LongCat-Image-Edit"
    prompt = """把图像中的人物转换成写实风格"""
    base_embedding_path = "z_longcat_embedding.pt"
    
    
    image_dir = r"F:\ImageSet\general_editing\reverse_style_test\longcat"
    image_list = [
        f"{image_dir}/1.png",
        f"{image_dir}/2.png",
        f"{image_dir}/3.png"
    ]
    embedding_list = []
    
    seed = 42

    timings = {}  # Store timings for each stage

    # ============ Stage 1: Encode Prompt (or load cached) ============
    text_processor = AutoProcessor.from_pretrained( model_path, subfolder = 'tokenizer'  )
    text_pipeline = None
    for idx, image_path in enumerate(image_list):
        base_embedding_path = f"z_longcat_embedding_{idx}.pt"
        if not os.path.exists(base_embedding_path):
            text_pipeline = LongCatImageEditPipeline.from_pretrained(
                model_path,
                transformer=None,
                text_processor=text_processor,
                vae=None,
                torch_dtype=dtype,
            ).to(device)
            break
    
    
    for idx, image_path in enumerate(image_list):
        base_embedding_path = f"z_longcat_embedding_{idx}.pt"
        start_time = time.time()
        if os.path.exists(base_embedding_path):
            print("Loading cached prompt embeddings...")
            base_embedding = torch.load(base_embedding_path)
            # prompt_embeds = base_embedding["prompt_embeds"]
            # text_ids = base_embedding["text_ids"]
            embedding_list.append(base_embedding)
        else:
            print("Encoding prompt...")
            prompt_image = crop_image(image_path, 512)
            # convert nparray to pil image and save to subdir
            prompt_image = Image.fromarray((prompt_image).astype('uint8'))
            ref_image = crop_image(image_path, 1024)
            # convert nparray to pil image and save to subdir
            ref_image = Image.fromarray((ref_image).astype('uint8'))
            with torch.no_grad():
                prompt_embeds,  text_ids = text_pipeline.encode_prompt(
                        [prompt_image],
                        [prompt],
                        device,
                        dtype)
            
            ref_image_path = f"z_{idx}_1024.png"
            # save image
            ref_image.save(ref_image_path)
            base_embedding = {
                "prompt_embeds": prompt_embeds,
                "text_ids": text_ids,
                "ref_image_path": ref_image_path,
            }
            torch.save(base_embedding, base_embedding_path)
            embedding_list.append(base_embedding)
            del base_embedding
            del prompt_image, ref_image
            flush()
    timings["Prompt Encoding"] = time.time() - start_time
    if text_pipeline is not None:  # Free up memory
        del text_pipeline

    # ============ Stage 2: Load Full Pipeline ============
    start_time = time.time()
    print("Loading ZImage pipeline...")
    pipe = LongCatImageEditPipeline.from_pretrained(
        model_path,
        transformer=None,
        text_encoder=None,
        tokenizer=None,
        text_processor=text_processor,
        torch_dtype=dtype,
    ).to(device)
    
    # load state dict from safetensors
    transformer_folder = os.path.join(model_path, "transformer")
    pipe.transformer = LongCatImageTransformer2DModel.from_pretrained(
        transformer_folder, 
        torch_dtype=dtype,
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
        ("lc_anything2real-4-2800", ["lc_anything2real-4-2800"], [1.0], step),
        ("lc_anything2real-5-3360", ["lc_anything2real-5-3360"], [1.0], step),
        ("lc_anything2real-6-3920", ["lc_anything2real-6-3920"], [1.0], step),
        ("lc_anything2real-7-4480", ["lc_anything2real-7-4480"], [1.0], step),
        ("lc_anything2real-8-5040", ["lc_anything2real-8-5040"], [1.0], step),
        ("lc_anything2real-9-5600", ["lc_anything2real-9-5600"], [1.0], step),
        
    ]
    
    def generate_image_with_lora(e_idx, idx, embedding, weight_name, lora_list, weight_list, steps):
        generator = torch.Generator(device=device).manual_seed(seed)
        prompt_embeds = embedding["prompt_embeds"]
        text_ids = embedding["text_ids"]
        ref_image_path = embedding["ref_image_path"]
        ref_image = Image.open(ref_image_path)
        output_path = f"z_edit_{e_idx}_{idx}_{weight_name}_{steps}.png"
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
            pipe.set_adapters(lora_list, adapter_weights=weight_list)

        start_time = time.time()
        print("Generating image with LoRA...")
        
        # steps = 9
        image = pipe(
            image=[ref_image],
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            num_inference_steps=steps,
            generator=generator,
            guidance_scale=2.5,
            num_images_per_prompt=1,
            # enable_cfg_renorm=True,
            # enable_prompt_rewrite=False  # Reusing the text encoder as a built-in prompt rewriter
        ).images[0]
        generation_time = time.time() - start_time

        pipe.unload_lora_weights()
        # Save result
        image.save(output_path)
        return generation_time
    
    for e_idx, embedding in enumerate(embedding_list):
        for idx, (lora_name, lora_list, weight_list, steps) in enumerate(loras):
            timings[f"LoRA {lora_name}"] = generate_image_with_lora(e_idx, idx, embedding, lora_name, lora_list, weight_list, steps)
        # del prompt_embeds, text_ids
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