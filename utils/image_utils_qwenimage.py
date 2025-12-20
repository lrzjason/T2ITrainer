from torch.utils.data import Dataset, Sampler
import random
import json
import torch
import os
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
# BASE_RESOLUTION = 1024

# RESOLUTION_SET = [
#     (1024, 1024),
#     (1152, 896),
#     (1216, 832),
#     (1344, 768),
#     (1536, 640),
# ]

RESOLUTION_CONFIG = {
    2048: [
        (1024, 2560),   # 1 ← 512x1280
        (1152, 2816),   # 2 ← 576x1408
        (1280, 3072),   # 3 ← 640x1536
        (1280, 2816),   # 4 ← 640x1408
        (1408, 3008),   # 5 ← 704x1504 → 3008 ÷ 64 = 47 ✅
        (1536, 2816),   # 6 ← 768x1408
        (1664, 2688),   # 7 ← 832x1344
        (1792, 2368),   # 8 ← 896x1184
        (1920, 2240),   # 9 ← 960x1120
        (2048, 2560),   # 10 ← 1024x1280
        (2176, 2304),   # 11 ← 1088x1152
        (2048, 2048),   # 12 ← 1024x1024
        (2048, 2304),   # 13 ← 1024x1152
    ],
    1536: [
        (768, 1920),    # 1 ← 512x1280
        (896, 2112),    # 2 ← 576x1408
        (960, 2304),    # 3 ← 640x1536
        (960, 2112),    # 4 ← 640x1408
        (1056, 2240),   # 5 ← 704x1504
        (1152, 2112),   # 6 ← 768x1408
        (1248, 2048),   # 7 ← 832x1344
        (1344, 1792),   # 8 ← 896x1184
        (1408, 1664),   # 9 ← 960x1120 ← FIXED: was (1440,1664)
        (1536, 1920),   # 10 ← 1024x1280
        (1632, 1728),   # 11 ← 1088x1152
        (1536, 1536),   # 12 ← 1024x1024
        (1536, 1728),   # 13 ← 1024x1152
    ],
    1328: [
        (640, 1664),    # 1 ← 512x1280
        (768, 1856),    # 2 ← 576x1408
        (832, 1984),    # 3 ← 640x1536
        (832, 1856),    # 4 ← 640x1408
        (896, 1920),    # 5 ← 704x1504
        (1024, 1856),   # 6 ← 768x1408
        (1088, 1728),   # 7 ← 832x1344
        (1152, 1536),   # 8 ← 896x1184
        (1280, 1472),   # 9 ← 960x1120 ← FIXED: was (1216,1472)
        (1344, 1664),   # 10 ← 1024x1280
        (1408, 1472),   # 11 ← 1088x1152
        (1344, 1344),   # 12 ← 1024x1024
        (1344, 1472),   # 13 ← 1024x1152
    ],
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
        (384, 960),     # 1 ← 512x1280
        (448, 1056),    # 2 ← 576x1408
        (512, 1152),    # 3 ← 640x1536
        (512, 1056),    # 4 ← 640x1408
        (512, 1088),    # 5 ← 704x1504 ← adjusted to avoid dup
        (576, 1056),    # 6 ← 768x1408
        (640, 1024),    # 7 ← 832x1344
        (672, 896),     # 8 ← 896x1184
        (704, 832),     # 9 ← 960x1120
        (768, 960),     # 10 ← 1024x1280
        (832, 896),     # 11 ← 1088x1152
        (768, 768),     # 12 ← 1024x1024
        (768, 832),     # 13 ← 1024x1152 ← adjusted to avoid dup
    ],
    512: [
        (256, 640),     # 1 ← 512x1280
        (320, 704),     # 2 ← 576x1408
        (320, 768),     # 3 ← 640x1536
        (320, 640),     # 4 ← 640x1408 ← adjusted to avoid dup
        (384, 768),     # 5 ← 704x1504
        (384, 704),     # 6 ← 768x1408
        (448, 704),     # 7 ← 832x1344 ← FIXED: was (448,672)
        (448, 576),     # 8 ← 896x1184
        (512, 576),     # 9 ← 960x1120
        (512, 640),     # 10 ← 1024x1280
        (576, 576),     # 11 ← 1088x1152
        (512, 512),     # 12 ← 1024x1024
        (576, 640),     # 13 ← 1024x1152 ← adjusted to avoid dup
    ],
    256: [
        (128, 320),
        (160, 352),
        (160, 384),
        (160, 320),
        (192, 384),
        (192, 352),
        (224, 352),
        (224, 288),
        (256, 288),
        (256, 320),
        (288, 288),
        (256, 256),
        (288, 320),
    ],
    128: [
        (64, 160),
        (80, 176),
        (80, 192),
        (80, 160),
        (96, 192),
        (96, 176),
        (112, 176),
        (112, 144),
        (128, 144),
        (128, 160),
        (144, 144),
        (128, 128),
        (144, 160),
    ],
    384: [
        (192, 480),     # 1
        (240, 528),     # 2
        (240, 576),     # 3
        (240, 480),     # 4
        (288, 576),     # 5
        (288, 528),     # 6
        (336, 528),     # 7
        (336, 432),     # 8
        (384, 432),     # 9
        (384, 480),     # 10
        (432, 432),     # 11
        (384, 384),     # 12
        (432, 480),     # 13
    ]
}


# PREFERRED_KONTEXT_RESOLUTIONS = [
#     (672, 1568),
#     (688, 1504),
#     (720, 1456),
#     (752, 1392),
#     (800, 1328),
#     (832, 1248),
#     (880, 1184),
#     (944, 1104),
#     (1024, 1024),
#     (1104, 944),
#     (1184, 880),
#     (1248, 832),
#     (1328, 800),
#     (1392, 752),
#     (1456, 720),
#     (1504, 688),
#     (1568, 672),
# ]


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

def closest_mod_64(value):
    return value - (value % 64)

# return closest_ratio and width,height closest_resolution
def get_nearest_resolution(image, resolution=1024):
    # height, width, _ = image.shape
    resolution_set = RESOLUTION_CONFIG[resolution]
    
    # # get ratio
    # image_ratio = width / height

    # target_set = resolution_set.copy()
    # reversed_set = [(y, x) for x, y in target_set]
    # target_set = sorted(set(target_set + reversed_set))
    # target_ratio = sorted(set([round(width/height, 2) for width,height in target_set]))
    
    # # Find the closest vertical ratio
    # closest_ratio = min(target_ratio, key=lambda x: abs(x - image_ratio))
    # closest_resolution = target_set[target_ratio.index(closest_ratio)]

    closest_ratio,closest_resolution = get_nearest_resolution_utils(image, resolution_set)

    return closest_ratio,closest_resolution

def crop_image(image_path, resolution=1024):
    resolution_set = RESOLUTION_CONFIG[resolution]
    image = crop_image_utils(image_path,resolution_set)
    return image


def get_latent(latent_path,latent_key, vae_config_shift_factor, vae_config_scaling_factor, device, weight_dtype):
    cached_latent = torch.load(latent_path, weights_only=True)
    latent = cached_latent[latent_key].to(device=device,dtype=weight_dtype)
    latent = (latent - vae_config_shift_factor) * vae_config_scaling_factor
    latent = latent.to(device=device,dtype=weight_dtype)
    # print("latent.shape",latent.shape)
    # latent = latent.unsqueeze(1)
    # [B, C, T=1, H, W]
    return latent

# def get_actual_emb(emb, prompt_embed_length):
#     if emb.ndim > 2:
#         sliced = [emb[i, :l, :] for i, l in enumerate([prompt_embed_length])]
#     else:
#         sliced = [emb[i, :l] for i, l in enumerate([prompt_embed_length])]
#     return pad_sequence(sliced, batch_first=True, padding_value=0)

def get_actual_emb(emb, prompt_embed_length):
    if emb.ndim > 2:
        sliced = [emb[i, :l, :] for i, l in enumerate([prompt_embed_length])]
    else:
        sliced = [emb[i, :l] for i, l in enumerate([prompt_embed_length])]
    return pad_sequence(sliced, batch_first=True, padding_value=0)

def get_caption_embedding(npz_path, device, weight_dtype, prompt_embed_key, prompt_embeds_mask_key, prompt_embed_length_key):
    cached_npz = torch.load(npz_path, weights_only=True)
    prompt_embed = torch.stack([cached_npz[prompt_embed_key].to(device=device,dtype=weight_dtype)])
    prompt_embeds_mask = torch.stack([cached_npz[prompt_embeds_mask_key].to(device=device,dtype=weight_dtype)])
    prompt_embed_length = torch.stack([cached_npz[prompt_embed_length_key]]).to(device=device)
    
    # print("prompt_embed_length", prompt_embed_length)
    prompt_embed = get_actual_emb(prompt_embed, prompt_embed_length)
    prompt_embeds_mask = get_actual_emb(prompt_embeds_mask, prompt_embed_length)
    
    
    return prompt_embed, prompt_embeds_mask


class ImagePairsDataset(Dataset):
    def __init__(self, datarows):
        # self.has_redux = has_redux
        self.datarows = datarows
        self.leftover_indices = []  #initialize an empty list to store indices of leftover items
    
        #returns dataset length
    def __len__(self):
        return len(self.datarows)
    #returns dataset item, using index
    def __getitem__(self, index):
        if self.leftover_indices:
            # Fetch from leftovers first
            actual_index = self.leftover_indices.pop(0)
        else:
            actual_index = index
        return self.datarows[actual_index] 

class CachedJsonDataset(Dataset):
    def __init__(self, datarows, 
                 vae_config_shift_factor, vae_config_scaling_factor, device, weight_dtype,
                 latent_path_key, latent_key, npz_path_key, prompt_embed_key, prompt_embeds_mask_key, prompt_embed_length_key,
                 from_latent_path_key, from_latent_key): 
        # self.has_redux = has_redux
        self.datarows = datarows
        self.leftover_indices = []  #initialize an empty list to store indices of leftover items
        # self.dataset_configs = dataset_configs
        self.empty_embedding = get_empty_embedding()
        
        
        self.vae_config_shift_factor = vae_config_shift_factor
        self.vae_config_scaling_factor = vae_config_scaling_factor
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
    #returns dataset length
    def __len__(self):
        return len(self.datarows)
    #returns dataset item, using index
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
                                item[self.latent_key] = get_latent(item[self.latent_path_key], self.latent_key, self.vae_config_shift_factor, self.vae_config_scaling_factor, self.device, self.weight_dtype)
                                if self.from_latent_path_key in item:
                                    item[self.from_latent_key] = get_latent(item[self.from_latent_path_key], self.latent_key, self.vae_config_shift_factor, self.vae_config_scaling_factor, self.device, self.weight_dtype)
                            
                    elif isinstance(configs, dict):
                        if self.npz_path_key in configs:
                            item = configs
                            item[self.prompt_embed_key], item[self.prompt_embeds_mask_key] = get_caption_embedding(item[self.npz_path_key], self.device, self.weight_dtype, self.prompt_embed_key, self.prompt_embeds_mask_key, self.prompt_embed_length_key)

        json_obj["dataset"] = path_obj["dataset"]
        return json_obj

class CachedMutiImageDataset(Dataset):
    def __init__(self, datarows,conditional_dropout_percent=0.1, has_redux=False, dataset_configs=None): 
        self.has_redux = has_redux
        self.datarows = datarows
        self.leftover_indices = []  #initialize an empty list to store indices of leftover items
        #for conditional_dropout
        self.conditional_dropout_percent = conditional_dropout_percent
        self.dataset_configs = dataset_configs
        self.empty_embedding = get_empty_embedding()
    #returns dataset length
    def __len__(self):
        return len(self.datarows)
    #returns dataset item, using index
    def __getitem__(self, index):
        if self.leftover_indices:
            # Fetch from leftovers first
            actual_index = self.leftover_indices.pop(0)
        else:
            actual_index = index
        metadata = self.datarows[actual_index] 
        result = {
            
        }
        metadata_caption_key = self.dataset_configs["caption_key"]
        npz_path_key = self.dataset_configs["npz_path_key"]
        npz_keys = self.dataset_configs["npz_keys"]
        latent_path_key = self.dataset_configs["latent_path_key"]
        latent_key = self.dataset_configs["latent_key"]
        keys = metadata.keys()
        for key in keys:
            if key == "mapping_key":
                result["mapping_key"] = metadata[key]
            if metadata_caption_key == key:
                captions = metadata[metadata_caption_key]
                for caption_key in captions.keys():
                    result[caption_key] = {
                        key:{}
                    }
                    caption = captions[caption_key]
                    cached_npz = torch.load(caption[npz_path_key], weights_only=True)
                    for npz_key in npz_keys:
                        result[caption_key][key][npz_key] = cached_npz[npz_key]
  
            if latent_path_key in metadata[key]:
                latent = torch.load(metadata[key][latent_path_key], weights_only=True)
                # for captions
                if key in result:
                    result[key][latent_key] = latent[latent_key]
                else:
                    result[key] = {
                        latent_key:latent[latent_key]
                    }
        return result

def compute_text_embeddings(text_encoders, tokenizers, prompt, device, image=None, processor=None, instruction=None, drop_index=None):
    with torch.no_grad():
        prompt_embeds, prompt_embeds_mask, prompt_embed_length, return_drop_idx = encode_prompt(text_encoders, 
                        tokenizers, prompt, device=device, image=image, processor=processor, 
                        instruction=instruction,
                        drop_index=drop_index)
        prompt_embeds = prompt_embeds.to(device)
        # text_ids = text_ids.to(device)
    return prompt_embeds, prompt_embeds_mask, prompt_embed_length, return_drop_idx

def vae_encode(vae,image):
    return vae_encode_utils(vae,image)

def get_empty_embedding(cache_path="cache/empty_embedding.npqwen"):
    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=True)
    else:
        raise FileNotFoundError(f"{cache_path} not found")
def create_empty_embedding(tokenizers,text_encoders,cache_path="cache/empty_embedding.npqwen",recreate=False):
    if recreate:
        os.remove(cache_path)

    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=True)

    prompt_embeds, prompt_embeds_mask, prompt_embed_length, _ = encode_prompt(text_encoders,tokenizers,"")
    prompt_embed = prompt_embeds.squeeze(0)
    prompt_embeds_mask = prompt_embeds_mask.squeeze(0)
    
    latent = {
        "prompt_embed": prompt_embed.cpu(), 
        "prompt_embeds_mask": prompt_embeds_mask.cpu(),
        "prompt_embed_length": prompt_embed_length
    }
    # save latent to cache file
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(latent, cache_path)

    return latent

def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

def extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

    return split_result
# reference: QwenImagePipeline _get_qwen_prompt_embeds
def encode_prompt_with_qwenvl(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    device=None,
    image=None,
    processor=None,
    instruction=None,
    drop_index=None,
):
    device = text_encoder.device
    dtype = text_encoder.dtype
    
    prompt = [prompt] if isinstance(prompt, str) else prompt
    
    template_prefix = "<|im_start|>system\n"
    instruction_content = None
    if instruction is not None:
        instruction_content = instruction
    template_suffix = "\n<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    if image is None or len(image) == 0:
        if instruction is None:
            instruction_content = "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:"
            drop_index = 34
        template_first_part = template_prefix + instruction_content
        
        if drop_index is None:
            model_inputs = tokenizer(
                template_first_part, max_length=max_sequence_length, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            _, drop_index = model_inputs.input_ids.shape
        
        template = template_first_part + template_suffix
            
        txt = [template.format(e) for e in prompt]
        
        model_inputs = tokenizer(
            txt, max_length=max_sequence_length + drop_index, padding=True, truncation=True, return_tensors="pt"
        ).to(device) 
        
        outputs = text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            output_hidden_states=True,
        )
    else:
        if instruction is None:
            instruction_content = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
            drop_index = 64
        template_first_part = template_prefix + instruction_content
        
        if drop_index is None:
            model_inputs = tokenizer(
                template_first_part, max_length=max_sequence_length, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            _, drop_index = model_inputs.input_ids.shape
            
        template = template_first_part + template_suffix
        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        if isinstance(image, list):
            base_img_prompt = ""
            for i, img in enumerate(image):
                base_img_prompt += img_prompt_template.format(i + 1)
        elif image is not None:
            base_img_prompt = img_prompt_template.format(1)
        else:
            base_img_prompt = ""
            
        txt = [template.format(base_img_prompt + e) for e in prompt]
        
        model_inputs = processor(
            text=txt,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        outputs = text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw.cpu(),
            output_hidden_states=True,
        )
       
    hidden_states = outputs.hidden_states[-1]
    split_hidden_states = extract_masked_hidden(hidden_states, model_inputs.attention_mask)
    split_hidden_states = [e[drop_index:] for e in split_hidden_states]
    attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
    prompt_embed_length = max([e.size(0) for e in split_hidden_states])
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(prompt_embed_length - u.size(0), u.size(1))]) for u in split_hidden_states]
    )
    encoder_attention_mask = torch.stack(
        [torch.cat([u, u.new_zeros(prompt_embed_length - u.size(0))]) for u in attn_mask_list]
    )
    
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # for stacking prompt embeds more than 1 bs
    if prompt_embeds.size(1) > max_sequence_length:
        prompt_embeds = prompt_embeds[:, :max_sequence_length]
    else:
        padding = torch.zeros(1, max_sequence_length - prompt_embeds.size(1), prompt_embeds.size(2), device=device)
        prompt_embeds = torch.cat([prompt_embeds, padding], dim=1)
    
    if encoder_attention_mask.size(1) > max_sequence_length:
        encoder_attention_mask = encoder_attention_mask[:, :max_sequence_length]
    else:
        mask_padding = torch.zeros(1, max_sequence_length - encoder_attention_mask.size(1), device=device)
        encoder_attention_mask = torch.cat([encoder_attention_mask, mask_padding], dim=1)
    
    
    prompt_embed_length = torch.tensor(prompt_embed_length).to(device)
    
    return prompt_embeds, encoder_attention_mask, prompt_embed_length, drop_index

def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length=1024,
    device=None,
    image=None,
    processor=None,
    instruction=None,
    drop_index=None
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds, prompt_embeds_mask, prompt_embed_length, return_drop_index = encode_prompt_with_qwenvl(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        device=device if device is not None else text_encoders[-1].device,
        image=image,
        processor=processor,
        instruction=instruction,
        drop_index=drop_index
    )
    return prompt_embeds, prompt_embeds_mask, prompt_embed_length, return_drop_index
    
