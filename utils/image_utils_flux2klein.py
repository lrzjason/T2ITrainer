from torch.utils.data import Dataset, Sampler
import random
import json
import torch
import os
from typing import Optional, List, Union, Tuple
from torchvision import transforms
from PIL import Image, ImageOps
from tqdm import tqdm 
import cv2
import numpy
from utils.utils import (
    replace_non_utf8_characters,
    get_md5_by_path,
    resize
)
import glob
from utils.dist_utils import flush
import numpy as np
import pandas as pd
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import load_image

from utils.utils import ToTensorUniversal, get_nearest_resolution_utils, crop_image_utils, vae_encode_utils

from torch.nn.utils.rnn import pad_sequence

from flux2klein.pipeline_flux2_klein import Flux2KleinPipeline

# Resolution configurations for Flux2-Klein based on FLUX.2 capabilities
RESOLUTION_CONFIG = {
    1024: [
        (512, 1280),    # 1
        (576, 1408),    # 2
        (640, 1536),    # 3
        (640, 1408),    # 4
        (704, 1504),    # 5
        (768, 1408),    # 6
        (832, 1344),    # 7
        (896, 1184),    # 8
        (960, 1120),    # 9
        (1024, 1280),   # 10
        (1088, 1152),   # 11
        (1024, 1024),   # 12
        (1024, 1152),   # 13
    ],
    768: [
        (384, 960),     # 1
        (448, 1056),    # 2
        (512, 1152),    # 3
        (512, 1056),    # 4
        (512, 1088),    # 5
        (576, 1056),    # 6
        (640, 1024),    # 7
        (672, 896),     # 8
        (704, 832),     # 9
        (768, 960),     # 10
        (832, 896),     # 11
        (768, 768),     # 12
        (768, 832),     # 13
    ],
    512: [
        (256, 640),     # 1
        (320, 704),     # 2
        (320, 768),     # 3
        (320, 640),     # 4
        (384, 768),     # 5
        (384, 704),     # 6
        (448, 704),     # 7
        (448, 576),     # 8
        (512, 576),     # 9
        (512, 640),     # 10
        (576, 576),     # 11
        (512, 512),     # 12
        (576, 640),     # 13
    ],
    2048: [
        (1024, 2560),   # 1
        (1152, 2816),   # 2
        (1280, 3072),   # 3
        (1280, 2816),   # 4
        (1408, 3008),   # 5
        (1536, 2816),   # 6
        (1664, 2688),   # 7
        (1792, 2368),   # 8
        (1920, 2240),   # 9
        (2048, 2560),   # 10
        (2176, 2304),   # 11
        (2048, 2048),   # 12
        (2048, 2304),   # 13
    ],
    1536: [
        (768, 1920),    # 1
        (896, 2112),    # 2
        (960, 2304),    # 3
        (960, 2112),    # 4
        (1056, 2240),   # 5
        (1152, 2112),   # 6
        (1248, 2048),   # 7
        (1344, 1792),   # 8
        (1408, 1664),   # 9
        (1536, 1920),   # 10
        (1632, 1728),   # 11
        (1536, 1536),   # 12
        (1536, 1728),   # 13
    ],
}


def get_buckets(resolution=1024):
    resolution_set = RESOLUTION_CONFIG[resolution]
    horizontal_resolution_set = resolution_set
    vertical_resolution_set = [(height,width) for width,height in resolution_set]
    all_resolution_set = horizontal_resolution_set + vertical_resolution_set
    # reduce duplicated res
    all_resolution_set = pd.Series(all_resolution_set).drop_duplicates().tolist()
    buckets = {}
    for resolution in all_resolution_set:
        buckets[f'{resolution[0]}x{resolution[1]}'] = []
    return buckets


# return closest_ratio and width,height closest_resolution
def get_nearest_resolution(image, resolution=1024):
    resolution_set = RESOLUTION_CONFIG[resolution]
    
    closest_ratio, closest_resolution = get_nearest_resolution_utils(image, resolution_set)

    return closest_ratio, closest_resolution


def crop_image(image_path, resolution=1024):
    resolution_set = RESOLUTION_CONFIG[resolution]
    image = crop_image_utils(image_path, resolution_set)
    return image


def get_latent(latent_path, latent_key, latents_bn_mean, latents_bn_std, device, weight_dtype):
    cached_latent = torch.load(latent_path, weights_only=True)
    latent = cached_latent[latent_key].to(device=device, dtype=weight_dtype)
    return latent


def get_actual_emb(emb, prompt_embed_length):
    if emb.ndim > 2:
        sliced = [emb[i, :l, :] for i, l in enumerate([prompt_embed_length])]
    else:
        sliced = [emb[i, :l] for i, l in enumerate([prompt_embed_length])]
    return pad_sequence(sliced, batch_first=True, padding_value=0)


def get_caption_embedding(npz_path, device, weight_dtype, prompt_embed_key, prompt_embeds_mask_key, prompt_embed_length_key):
    cached_npz = torch.load(npz_path, weights_only=True)
    prompt_embed = torch.stack([cached_npz[prompt_embed_key].to(device=device, dtype=weight_dtype)])
    prompt_embeds_mask = torch.stack([cached_npz[prompt_embeds_mask_key].to(device=device, dtype=weight_dtype)])
    prompt_embed_length = torch.stack([cached_npz[prompt_embed_length_key]]).to(device=device)
    
    prompt_embed = get_actual_emb(prompt_embed, prompt_embed_length)
    prompt_embeds_mask = get_actual_emb(prompt_embeds_mask, prompt_embed_length)
    
    return prompt_embed, prompt_embeds_mask


class ImagePairsDataset(Dataset):
    def __init__(self, datarows):
        self.datarows = datarows
        self.leftover_indices = []  # initialize an empty list to store indices of leftover items
    
    # returns dataset length
    def __len__(self):
        return len(self.datarows)
        
    # returns dataset item, using index
    def __getitem__(self, index):
        if self.leftover_indices:
            # Fetch from leftovers first
            actual_index = self.leftover_indices.pop(0)
        else:
            actual_index = index
        return self.datarows[actual_index] 


class CachedJsonDataset(Dataset):
    def __init__(self, datarows, 
                 latents_bn_mean, latents_bn_std, device, weight_dtype,
                 latent_path_key, latent_key, npz_path_key, prompt_embed_key, 
                 prompt_embeds_mask_key, prompt_embed_length_key,
                 from_latent_path_key, from_latent_key): 
        self.datarows = datarows
        self.leftover_indices = []  # initialize an empty list to store indices of leftover items
        self.empty_embedding = get_empty_embedding()
        
        
        self.latents_bn_mean = latents_bn_mean
        self.latents_bn_std = latents_bn_std
        self.device = device
        self.weight_dtype = weight_dtype
        
        self.latent_path_key = latent_path_key
        self.latent_key = latent_key
        
        self.npz_path_key = npz_path_key
        self.prompt_embed_key = prompt_embed_key
        self.prompt_embeds_mask_key = prompt_embeds_mask_key
        self.prompt_embed_length_key = prompt_embed_length_key
        self.from_latent_path_key = from_latent_path_key
        self.from_latent_key = from_latent_key
    
    # returns dataset length
    def __len__(self):
        return len(self.datarows)
        
    # returns dataset item, using index
    def __getitem__(self, index):
        if self.leftover_indices:
            # Fetch from leftovers first
            actual_index = self.leftover_indices.pop(0)
        else:
            actual_index = index
        path_obj = self.datarows[actual_index] 
        json_path = path_obj["json_path"]
        # load metadata
        with open(json_path, 'r', encoding='utf-8') as f:
            json_obj = json.load(f)
        
        for _, group_configs in json_obj.items():
            # check item is dict
            if isinstance(group_configs, dict):
                for _, configs in group_configs.items():
                    if isinstance(configs, list):
                        for item in configs:
                            if self.latent_path_key in item:
                                item[self.latent_key] = get_latent(item[self.latent_path_key], self.latent_key, self.latents_bn_mean, self.latents_bn_std, self.device, self.weight_dtype)
                                if self.from_latent_path_key in item:
                                    item[self.from_latent_key] = get_latent(item[self.from_latent_path_key], self.latent_key, self.latents_bn_mean, self.latents_bn_std, self.device, self.weight_dtype)
                            
                    elif isinstance(configs, dict):
                        if self.npz_path_key in configs:
                            item = configs
                            item[self.prompt_embed_key], item[self.prompt_embeds_mask_key] = get_caption_embedding(item[self.npz_path_key], self.device, self.weight_dtype, self.prompt_embed_key, self.prompt_embeds_mask_key, self.prompt_embed_length_key)

        json_obj["dataset"] = path_obj["dataset"]
        return json_obj

def compute_text_embeddings(text_encoders, tokenizers, prompt, device, image=None, processor=None, instruction=None, drop_index=None):
    with torch.no_grad():
        prompt_embeds, prompt_embeds_mask, prompt_embed_length, return_drop_idx = encode_prompt(text_encoders, 
                        tokenizers, prompt, device=device, image=image, processor=processor, 
                        instruction=instruction,
                        drop_index=drop_index)
        prompt_embeds = prompt_embeds.to(device)
    return prompt_embeds, prompt_embeds_mask, prompt_embed_length, return_drop_idx


def vae_encode(vae, image):
    return vae_encode_utils(vae, image, vae_type="flux")


def get_empty_embedding(cache_path="cache/empty_embedding.npf2k"):
    if os.path.exists(cache_path):
        return get_caption_embedding(cache_path, device='cpu', weight_dtype=torch.float32,
                              prompt_embed_key="prompt_embed", prompt_embeds_mask_key="prompt_embeds_mask",
                              prompt_embed_length_key="prompt_embed_length")
    else:
        raise FileNotFoundError(f"{cache_path} not found")

def create_empty_embedding(tokenizers, text_encoders, cache_path="cache/empty_embedding.npf2k", recreate=False):
    if recreate:
        os.remove(cache_path)

    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=True)

    prompt_embeds, prompt_embeds_mask, prompt_embed_length, _ = encode_prompt(text_encoders, tokenizers, "")
    prompt_embed = prompt_embeds.squeeze(0)
    prompt_embeds_mask = prompt_embeds_mask.squeeze(0)
    
    cache = {
        "prompt_embed": prompt_embed.cpu(), 
        "prompt_embeds_mask": prompt_embeds_mask.cpu(),
        "prompt_embed_length": prompt_embed_length
    }
    # save latent to cache file
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(cache, cache_path)

    return cache


def get_latent_config(cache_path="cache/latent_config.npf2k"):
    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=True)
    else:
        raise FileNotFoundError(f"{cache_path} not found")


def create_latent_config(vae, cache_path="cache/latent_config.npf2k"):
    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=True)

    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(vae.device)
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
        vae.device
    )
    latent = {
        "latents_bn_mean": latents_bn_mean.cpu(),
        "latents_bn_std": latents_bn_std.cpu()
    }
    # save latent to cache file
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(latent, cache_path)

    return latent

def extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

    return split_result


def encode_prompt_with_qwen3(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt="",
    device=None,
    hidden_states_layers=(9, 18, 27),
):
    """
    Encode prompt using Qwen3ForCausalLM for Flux2-Klein pipeline
    """
    device = device or text_encoder.device
    dtype = text_encoder.dtype
    
    prompt = [prompt] if isinstance(prompt, str) else prompt

    all_input_ids = []
    all_attention_masks = []

    for single_prompt in prompt:
        messages = [{"role": "user", "content": single_prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )

        all_input_ids.append(inputs["input_ids"])
        all_attention_masks.append(inputs["attention_mask"])

    input_ids = torch.cat(all_input_ids, dim=0).to(device)
    attention_mask = torch.cat(all_attention_masks, dim=0).to(device)

    # Forward pass through the model
    output = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )

    # Only use outputs from intermediate layers and stack them
    out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
    out = out.to(dtype=dtype, device=device)

    batch_size, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length=512,
    text_encoder_out_layers=(9, 18, 27),
    device=None,
    image=None,  # Added for compatibility
    processor=None,  # Added for compatibility
    instruction=None,  # Added for compatibility
    drop_index=None,  # Added for compatibility
):
    """
    Encode prompt for Flux2-Klein pipeline
    """
    prompt = [prompt] if isinstance(prompt, str) else prompt
    
    # Use the last text encoder and tokenizer (Qwen3 for Flux2-Klein)
    prompt_embeds = encode_prompt_with_qwen3(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        device=device if device is not None else text_encoders[-1].device,
        hidden_states_layers=text_encoder_out_layers
    )
    
    # Create attention mask based on non-padding tokens
    attention_mask = torch.ones_like(prompt_embeds[..., 0]).long()
    
    # Calculate prompt length (this is simplified since we're using fixed-length padding)
    prompt_embed_length = torch.tensor(prompt_embeds.size(1)).to(prompt_embeds.device)
    
    return prompt_embeds, attention_mask, prompt_embed_length, None
