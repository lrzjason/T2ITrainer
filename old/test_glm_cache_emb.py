import torch
import time
from diffusers.pipelines.glm_image import GlmImagePipeline
from PIL import Image
import gc
import os
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

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
    
    # Define paths for image-to-image
    # image_path = "glm_cached_embeddings_output_en_1536.png"  # Path to input conditional image
    # width = 32 * 32  # Width for image-to-image (as in test_glm.py)
    # height = 33 * 32  # Height for image-to-image (as in test_glm.py)
    width = 1024
    height = 1024
    
    device = torch.device("cuda")
    dtype = torch.bfloat16
    model_path = r"F:\\HF_Models\\GLM\\GLM-Image"
    # Example prompt for image-to-image
    # prompt = "Replace the background of the snow forest with an underground station featuring an automatic escalator."
    prompt = "A beautifully designed modern food magazine style dessert recipe illustration, themed around a raspberry mousse cake. The overall layout is clean and bright, divided into four main areas: the top left features a bold black title 'Raspberry Mousse Cake Recipe Guide', with a soft-lit close-up photo of the finished cake on the right, showcasing a light pink cake adorned with fresh raspberries and mint leaves; the bottom left contains an ingredient list section, titled 'Ingredients' in a simple font, listing 'Flour 150g', 'Eggs 3', 'Sugar 120g', 'Raspberry puree 200g', 'Gelatin sheets 10g', 'Whipping cream 300ml', and 'Fresh raspberries', each accompanied by minimalist line icons (like a flour bag, eggs, sugar jar, etc.); the bottom right displays four equally sized step boxes, each containing high-definition macro photos and corresponding instructions, arranged from top to bottom as follows: Step 1 shows a whisk whipping white foam (with the instruction 'Whip egg whites to stiff peaks'), Step 2 shows a red-and-white mixture being folded with a spatula (with the instruction 'Gently fold in the puree and batter'), Step 3 shows pink liquid being poured into a round mold (with the instruction 'Pour into mold and chill for 4 hours'), Step 4 shows the finished cake decorated with raspberries and mint leaves (with the instruction 'Decorate with raspberries and mint'); a light brown information bar runs along the bottom edge, with icons on the left representing 'Preparation time: 30 minutes', 'Cooking time: 20 minutes', and 'Servings: 8'. The overall color scheme is dominated by creamy white and light pink, with a subtle paper texture in the background, featuring compact and orderly text and image layout with clear information hierarchy."
    base_embedding_path = "glm_base_embedding.pt"
    prior_tokens_path = "glm_prior_tokens.pt"
    
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
        text_pipeline = GlmImagePipeline.from_pretrained(
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
    # cond_image = Image.open(image_path).convert("RGB")
    cond_image = None
    cond_image_list = None
    if cond_image is not None:
        cond_image_list = [cond_image]
    
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
        text_pipeline = GlmImagePipeline.from_pretrained(
            model_path,
            text_encoder=None,
            vae=None,
            transformer=None,
            torch_dtype=dtype,
        ).to(device)
        
        with torch.no_grad():
            prior_token_ids, prior_image_token_ids = text_pipeline.generate_prior_tokens(
                prompt=prompt,
                image=cond_image_list,
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
    print("Loading GLM pipeline...")
    pipe = GlmImagePipeline.from_pretrained(
        model_path,
        text_encoder=None,
        vision_language_encoder=None,
        torch_dtype=dtype,
    ).to(device)
    
    pipe.enable_model_cpu_offload()

    timings["Pipeline Loading"] = time.time() - start_time
    
    

    # ============ Stage 3: Image Generation ============
    start_time = time.time()
    print("Generating image...")
    step = 50
    pipe = pipe.to(device, dtype)
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Generate image using cached embeddings and prior tokens
    image = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        prior_token_ids=prior_token_ids,
        prior_image_token_ids=prior_image_token_ids,
        image=cond_image_list,  # Conditional image for image-to-image
        height=height,
        width=width,
        num_inference_steps=step,
        guidance_scale=1.5,
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