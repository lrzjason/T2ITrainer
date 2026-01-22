import torch
import time
from glm_image.pipeline_glm_image_img2img import GlmImageImg2ImgPipeline
from PIL import Image
import gc
import os


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()


@torch.no_grad()
def main():
    # Define paths for image-to-image
    image_path = r"F:\ImageSet\ObjectRemoval_test\1.png"  # Path to input conditional image
    width = 1536
    height = 1536
    
    device = torch.device("cuda")
    dtype = torch.bfloat16
    model_path = r"F:\\HF_Models\\GLM\\GLM-Image"
    # Example prompt for image-to-image
    prompt = "Replace the background of the snow forest with an underground station featuring an automatic escalator."
    base_embedding_path = "glm_i2i_base_embedding.pt"
    prior_tokens_path = "glm_i2i_prior_tokens.pt"
    
    seed = 42

    timings = {}  # Store timings for each stage

    # ============ Stage 1: Encode Prompt (or load cached) ============
    start_time = time.time()
    if os.path.exists(base_embedding_path):
        print("Loading cached prompt embeddings...")
        base_embedding = torch.load(base_embedding_path)
        prompt_embeds = base_embedding["prompt_embeds"]
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
    else:
        print("Encoding prompt...")
        text_pipeline = GlmImageImg2ImgPipeline.from_pretrained(
            model_path,
            vision_language_encoder=None,
            transformer=None,
            vae=None,
            torch_dtype=dtype,
        ).to(device)
        
        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds = text_pipeline.encode_prompt(prompt)
        
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        base_embedding = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
        }
        torch.save(base_embedding, base_embedding_path)
        del text_pipeline
        flush()
    timings["Prompt Encoding"] = time.time() - start_time

    # Load conditional image for image-to-image
    cond_image = [Image.open(image_path).convert("RGB")]
    
    # ============ Stage 1b: Generate Prior Tokens (or load cached) ============
    start_time = time.time()
    if os.path.exists(prior_tokens_path):
        print("Loading cached prior tokens...")
        prior_tokens_data = torch.load(prior_tokens_path)
        prior_token_ids = prior_tokens_data["prior_token_ids"]
        prior_image_token_ids = prior_tokens_data.get("prior_image_token_ids", None)
    else:
        print("Generating prior tokens...")
        # We need the vision_language_encoder and processor for generating prior tokens
        # So we load a partial pipeline with these components
        text_pipeline = GlmImageImg2ImgPipeline.from_pretrained(
            model_path,
            text_encoder=None,
            vae=None,
            transformer=None,
            torch_dtype=dtype,
        ).to(device)
        
        with torch.no_grad():
            prior_token_ids, prior_image_token_ids = text_pipeline.generate_prior_tokens(
                prompt=prompt,
                image=cond_image,
                height=height,
                width=width,
                device=device,
            )
        
        prior_tokens_data = {
            "prior_token_ids": prior_token_ids,
            "prior_image_token_ids": prior_image_token_ids,
        }
        torch.save(prior_tokens_data, prior_tokens_path)
        del text_pipeline
        flush()
    timings["Prior Tokens Generation"] = time.time() - start_time

    # ============ Stage 2: Load Full Pipeline ============
    start_time = time.time()
    print("Loading GLM image-to-image pipeline...")
    pipe = GlmImageImg2ImgPipeline.from_pretrained(
        model_path,
        text_encoder=None,
        vision_language_encoder=None,
        torch_dtype=dtype,
    ).to(device)
    
    pipe.enable_model_cpu_offload()

    timings["Pipeline Loading"] = time.time() - start_time

    # ============ Stage 3: Image Generation ============
    start_time = time.time()
    print("Generating image with image-to-image pipeline...")
    step = 50
    pipe = pipe.to(device, dtype)
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Generate image using cached embeddings and prior tokens
    image = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        prior_token_ids=prior_token_ids,
        prior_image_token_ids=prior_image_token_ids,
        image=cond_image,  # Conditional image for image-to-image
        height=height,
        width=width,
        num_inference_steps=step,
        guidance_scale=1.5,
        strength=0.5,  # Strength for image-to-image transformation
        generator=generator,
    ).images[0]
    generation_time = time.time() - start_time
    timings["Image Generation"] = generation_time

    output_path = f"glm_i2i_cached_embeddings_output.png"
    # Save result
    image.save(output_path)
    print(f"Image saved to {output_path}")

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