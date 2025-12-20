import torch
from diffusers import FluxPipeline, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from typing import Any, Callable, Dict, List, Optional, Union
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
    
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor

import numpy as np
import inspect
import gc
from accelerate import Accelerator

# from transformer_flux_masked import MaskedFluxTransformer2DModel
from optimum.quanto import freeze, qfloat8, quantize
from tqdm import tqdm

from diffusers import FluxTransformer2DModel
# from transformer_flux_mspace import MSpaceFluxTransformer2DModel
from diffusers import FluxFillPipeline
# from pipeline_flux_mspace import MSpacePlusFluxPipeline
import os

from diffusers import FluxPriorReduxPipeline
from diffusers.utils import load_image

image = load_image("F:/ImageSet/ObjectRemoval/test.jpg")
mask = load_image("F:/ImageSet/ObjectRemoval/test_mask.png")
    
device = torch.device("cuda")
dtype = torch.bfloat16

prompt = "Remove the selected objects from the image."
flux_dir = "F:/T2ITrainer/flux_models/fill"

base_embedding_path = "0_base_embedding.pt"
modified_embedding_path = "0_modified_embedding.pt"
def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

if os.path.exists(base_embedding_path):
    base_embedding = torch.load(base_embedding_path)
    
    prompt_embeds = base_embedding["prompt_embeds"]
    pooled_prompt_embeds = base_embedding["pooled_prompt_embeds"]
    text_ids = base_embedding["text_ids"]
    # target_index_lists = base_embedding["target_index_lists"]
    
    # target_index_list = base_embedding["target_index_list"]
    # modified_pooled_prompt_embeds = modified_embedding["pooled_prompt_embeds"]
else:
    text_encoder = CLIPTextModel.from_pretrained(
        flux_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        flux_dir, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
    )
    tokenizer = CLIPTokenizer.from_pretrained(flux_dir, subfolder="tokenizer")
    tokenizer_2 = T5TokenizerFast.from_pretrained(flux_dir, subfolder="tokenizer_2")
    # 第一阶段：加载text_encoder 和 tokenizer处理prompt
    pipeline = FluxFillPipeline.from_pretrained(
        flux_dir,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        transformer=None,
        vae=None,
    ).to("cuda")
    
    with torch.no_grad():
        print("Encoding prompts.")
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt, prompt_2=None, max_sequence_length=512
        )
        base_embedding = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "text_ids": text_ids
        }
        # save to base_embedding
        torch.save(base_embedding, base_embedding_path)
        
    del text_encoder
    del text_encoder_2
    del tokenizer
    del tokenizer_2
    del pipeline
    flush()



pipeline = FluxFillPipeline.from_pretrained(
    flux_dir,
    transformer=None,
    text_encoder=None,
    text_encoder_2=None,
    tokenizer=None,
    tokenizer_2=None,
    torch_dtype=torch.bfloat16,
).to(device)

print('loading flux transformer……')
transformer = FluxTransformer2DModel.from_pretrained(
    flux_dir, 
    subfolder="transformer", # 下载的fp8模型地址
    torch_dtype=dtype,
    local_files_only=True
)
pipeline.transformer = transformer

# print("Loading lora")
# pipeline.load_lora_weights("F:/models/Lora/flux", weight_name="objectRemoval_rordtest_reg08-0-16314.safetensors", adapter_name="removal")
# pipeline.set_adapters(["removal"], adapter_weights=[1])
# print("Fusing lora")
# pipeline.fuse_lora()

print('loaded flux transformer')
print('optimized flux transformer')
quantize(transformer, weights=qfloat8) # 对模型进行量化
freeze(transformer)


pipeline.enable_model_cpu_offload()
# pipeline.transformer.enable_layer_wise_casting()
dtype = torch.bfloat16

image = pipeline(
    num_inference_steps=30,
    prompt_embeds=prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    image=image,
    # mask_image=mask,
    height=512,
    width=512,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save(f"output_removal.png")

# pipeline.load_lora_weights("F:/models/Lora/flux", weight_name="objectRemoval_rordtest_reg08-0-16314.safetensors", adapter_name="removal")
# pipeline.set_adapters("removal")

# image = pipeline(
#     num_inference_steps=30,
#     prompt_embeds=prompt_embeds,
#     pooled_prompt_embeds=pooled_prompt_embeds,
#     image=image,
#     mask_image=mask,
#     height=512,
#     width=512,
#     max_sequence_length=512,
#     cross_attention_kwargs={"scale": 1},
#     generator=torch.Generator("cpu").manual_seed(0)
# ).images[0]
# image.save(f"output_lora.png")
