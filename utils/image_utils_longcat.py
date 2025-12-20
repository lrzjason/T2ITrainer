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
from longcat.model_utils import split_quotation, prepare_pos_ids
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


def get_latent(latent_path, latent_key, vae_config_shift_factor, vae_config_scaling_factor, device, weight_dtype):
    cached_latent = torch.load(latent_path, weights_only=True)
    latent = cached_latent[latent_key].to(device=device,dtype=weight_dtype)
    latent = (latent - vae_config_shift_factor) * vae_config_scaling_factor
    latent = latent.to(device=device,dtype=weight_dtype)
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

def get_caption_embedding(npz_path, device, weight_dtype, prompt_embed_key, text_id_key):
    cached_npz = torch.load(npz_path, weights_only=True)
    prompt_embed = torch.stack([cached_npz[prompt_embed_key].to(device=device,dtype=weight_dtype)])
    text_id = torch.stack([cached_npz[text_id_key].to(device=device,dtype=weight_dtype)])
    
    return prompt_embed, text_id


class CachedJsonDataset(Dataset):
    def __init__(self, datarows, 
                 vae_config_shift_factor, vae_config_scaling_factor, device, weight_dtype,
                 latent_path_key, latent_key, npz_path_key, prompt_embed_key, text_id_key,
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
        self.text_id_key = text_id_key
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
                            item[self.prompt_embed_key], item[self.text_id_key] = get_caption_embedding(item[self.npz_path_key], self.device, self.weight_dtype, self.prompt_embed_key, self.text_id_key)

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
        prompt_embeds, text_ids = encode_prompt(text_encoders, 
                        tokenizers, prompt, device=device, image=image, processor=processor, 
                        instruction=instruction,
                        drop_index=drop_index)
        prompt_embeds = prompt_embeds.to(device)
        # text_ids = text_ids.to(device)
    return prompt_embeds, text_ids

def vae_encode(vae,image):
    return vae_encode_utils(vae,image)

def get_empty_embedding(cache_path="cache/empty_embedding.nplongcat"):
    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=True)
    else:
        raise FileNotFoundError(f"{cache_path} not found")
def create_empty_embedding(tokenizers,text_encoders,cache_path="cache/empty_embedding.nplongcat",recreate=False):
    if recreate:
        os.remove(cache_path)

    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=True)

    prompt_embeds, text_ids = encode_prompt(text_encoders,tokenizers,"")
    prompt_embed = prompt_embeds.squeeze(0)
    # text_id = text_ids.squeeze(0)
    # prompt_embeds_mask = prompt_embeds_mask.squeeze(0)
    
    latent = {
        "prompt_embed": prompt_embed.cpu(), 
        "text_id": text_ids.cpu()
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
    prompt = [p.strip('"') if p.startswith('"') and p.endswith('"') else p for p in prompt]
    
    all_tokens = []
    for clean_prompt_sub, matched in split_quotation(prompt[0]):
        if matched:
            for sub_word in clean_prompt_sub:
                tokens = tokenizer(sub_word, add_special_tokens=False)['input_ids']
                all_tokens.extend(tokens)
        else:
            tokens = tokenizer(clean_prompt_sub, add_special_tokens=False)['input_ids']
            all_tokens.extend(tokens)

    all_tokens = all_tokens[:max_sequence_length]
    
    text_tokens_and_mask = tokenizer.pad(
        {'input_ids': [all_tokens]},
        max_length=max_sequence_length,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt')
    if image is None or len(image) == 0:
        # prompt_template_encode_start_idx = 36
        # prompt_template_encode_end_idx = 5
        prompt_template_encode_prefix = '<|im_start|>system\nAs an image captioning expert, generate a descriptive text prompt based on an image content, suitable for input to a text-to-image model.<|im_end|>\n<|im_start|>user\n'
        prompt_template_encode_suffix = '<|im_end|>\n<|im_start|>assistant\n'
        text = prompt_template_encode_prefix
    else:
        image_processor_vl = processor.image_processor
        raw_vl_input = image_processor_vl(images=image,return_tensors="pt")
        pixel_values = raw_vl_input['pixel_values'].to(device, dtype=dtype)
        image_grid_thw = raw_vl_input['image_grid_thw']
        # prompt_template_encode_start_idx = 67
        # prompt_template_encode_end_idx = 5
        prompt_template_encode_prefix = "<|im_start|>system\nAs an image editing expert, first analyze the content and attributes of the input image(s). Then, based on the user's editing instructions, clearly and precisely determine how to modify the given image(s), ensuring that only the specified parts are altered and all other aspects remain consistent with the original(s).<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        prompt_template_encode_suffix = '<|im_end|>\n<|im_start|>assistant\n'
        text = prompt_template_encode_prefix
        merge_length = image_processor_vl.merge_size**2
        image_token = "<|image_pad|>"
        while image_token in text:
            num_image_tokens = image_grid_thw.prod() // merge_length
            text = text.replace(image_token, "<|placeholder|>" * num_image_tokens, 1)
        text = text.replace("<|placeholder|>", image_token)

    
    prefix_tokens = tokenizer(text, add_special_tokens=False)['input_ids']
    suffix_tokens = tokenizer(prompt_template_encode_suffix, add_special_tokens=False)['input_ids']
    prefix_len = len(prefix_tokens)
    suffix_len = len(suffix_tokens)
        
    prefix_tokens_mask = torch.tensor( [1]*len(prefix_tokens), dtype = text_tokens_and_mask.attention_mask[0].dtype )
    suffix_tokens_mask = torch.tensor( [1]*len(suffix_tokens), dtype = text_tokens_and_mask.attention_mask[0].dtype )
    
    prefix_tokens = torch.tensor(prefix_tokens,dtype=text_tokens_and_mask.input_ids.dtype)
    suffix_tokens = torch.tensor(suffix_tokens,dtype=text_tokens_and_mask.input_ids.dtype)
            
    input_ids = torch.cat( (prefix_tokens, text_tokens_and_mask.input_ids[0], suffix_tokens), dim=-1 )
    attention_mask = torch.cat( (prefix_tokens_mask, text_tokens_and_mask.attention_mask[0], suffix_tokens_mask), dim=-1 )

    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)

    if image is None or len(image) == 0:
        text_output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
    else:
        text_output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw =image_grid_thw,
            output_hidden_states=True
        )
        
    # [max_sequence_length, batch, hidden_size] -> [batch, max_sequence_length, hidden_size]
    # clone to have a contiguous tensor
    prompt_embeds = text_output.hidden_states[-1].detach()
    
    prompt_embeds = prompt_embeds[:,prefix_len: -suffix_len ,:]

    text_ids = prepare_pos_ids(modality_id=0,
                                type='text',
                                start=(0, 0),
                                num_token=prompt_embeds.shape[1]).to(device, dtype=dtype)
    
    return prompt_embeds, text_ids

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
    prompt_embeds, text_ids = encode_prompt_with_qwenvl(
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
    return prompt_embeds, text_ids
    
