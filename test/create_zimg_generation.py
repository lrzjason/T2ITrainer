import torch
import time
# from diffusers import ZImagePipeline
from z_img.ZImageLoraPipeline import ZImagePipeline
import gc
import os
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

from diffusers import ZImageTransformer2DModel

from tqdm import tqdm
def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

@torch.no_grad()
def main():
    width = 1024
    height = 1024

    device = torch.device("cuda")
    dtype = torch.bfloat16
    model_path = r"F:\T2ITrainer_pulic\T2ITrainer\z_img_models\Z-Image-Turbo"
    prompt = "a photo"
    output_dir = r"F:\ImageSet\zimage_cache"
    os.makedirs(output_dir, exist_ok=True)
    
    base_embedding_path = "zimage_base_embedding.pt"
    
    base_seed = 43
    seed_step = 10000
    # total_count = 1000
    total_count = 2

    timings = {}
    start_time = time.time()

    # ============ Stage 1: Encode Prompt (or load cached) ============
    print(">>> Stage 1: Prompt Encoding")
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
            # prompt_embeds = prompt_embeds.to(dtype=dtype)  # ensure dtype
            # prompt_embeds = prompt_embeds[0].to(dtype=dtype)
            
        base_embedding = {"prompt_embeds": prompt_embeds}
        torch.save(base_embedding, base_embedding_path)
        del text_pipeline
        flush()
    timings["Prompt Encoding"] = time.time() - start_time

    # ============ Stage 2: Load Full Pipeline ============
    start_time = time.time()
    print(">>> Stage 2: Pipeline Loading")
    print("Loading ZImage pipeline (without text components)...")
    pipe = ZImagePipeline.from_pretrained(
        model_path,
        text_encoder=None,
        tokenizer=None,
        transformer=None,
        torch_dtype=dtype,
    ).to(device)

    print("Loading & compiling transformer...")
    transformer = ZImageTransformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=dtype,
    ).to(device)
    # transformer = torch.compile(transformer)
    pipe.transformer = transformer

    # Optional: enable memory optimization (uncomment if OOM)
    pipe.enable_model_cpu_offload()

    timings["Pipeline Loading"] = time.time() - start_time

    # ============ Stage 3: Batch Image Generation ============
    print(f">>> Stage 3: Generating {total_count} images...")
    gen_start_time = time.time()

    # Warm-up run (first image, includes CUDA kernel launch & compilation overhead)
    # generator = torch.Generator(device=device).manual_seed(base_seed)
    # _ = pipe(
    #     prompt_embeds=prompt_embeds,
    #     height=height,
    #     width=width,
    #     num_inference_steps=8,
    #     guidance_scale=0.0,
    #     generator=generator,
    # ).images[0]
    # warmup_time = time.time() - gen_start_time
    # timings["First Image (Warm-up)"] = warmup_time

    # print(f"✅ Warm-up done in {warmup_time:.2f}s. Starting batch generation...")

    # Track per-image times (excluding first)
    gen_times = []
    for i in tqdm(range(total_count + 1)):
        output_path = f"zimage_{i}.png"
        
        output_path = f"{output_dir}/{output_path}"
        if os.path.exists(output_path):
            print(f"⏭️  Skipping {output_path} (already exists)")
            continue

        iter_start = time.time()
        generator = torch.Generator(device=device).manual_seed(base_seed + i - 1 + seed_step)  # seed_i = base_seed + i - 1
        # generator = torch.Generator(device=device)
        image = pipe(
            prompt_embeds=prompt_embeds,
            height=height,
            width=width,
            num_inference_steps=8,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]
        gen_time = time.time() - iter_start
        gen_times.append(gen_time)

        image.save(output_path)
        
        # save a text file to store with prompt
        with open(f"{output_dir}/zimage_{i}.txt", "w") as f:
            f.write(prompt)

        if i % 100 == 0 or i == 1 or i == total_count:
            avg_gen_time = sum(gen_times) / len(gen_times) if gen_times else 0
            print(f"[{i:4d}/{total_count}] Saved {output_path} | Time: {gen_time:.2f}s (avg: {avg_gen_time:.2f}s)")

    timings["Batch Generation (avg per image)"] = sum(gen_times) / len(gen_times) if gen_times else 0.0
    timings["Batch Generation (total)"] = sum(gen_times)
    # timings["Total Generation (incl. warm-up)"] = warmup_time + sum(gen_times)

    del pipe
    flush()

    # ============ Print Timing Summary ============
    print("\n" + "="*50)
    print("✅ TIMING SUMMARY")
    print("="*50)
    total_time = 0.0
    for stage, t in timings.items():
        if isinstance(t, float):
            print(f"{stage:30s}: {t:7.2f} seconds")
            if stage not in ["Prompt Encoding", "Pipeline Loading"]:  # generation stages
                total_time += t
        else:
            print(f"{stage:30s}: {t}")
    print("-"*50)
    print(f"{'Total (all stages)':30s}: {sum(timings.values()):7.2f} seconds")
    print(f"{'Total (gen only, 1000 imgs)':30s}: {timings.get('Total Generation (incl. warm-up)', 0):7.2f} seconds")
    if gen_times:
        print(f"{'→ Avg per image (post warm-up)':30s}: {sum(gen_times)/len(gen_times):7.2f} seconds")
        print(f"{'→ Throughput':30s}: {len(gen_times) / sum(gen_times):7.2f} img/s")
    print("="*50)

if __name__ == "__main__":
    main()