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

from utils.utils import ToTensorUniversal

# BASE_RESOLUTION = 1024

# RESOLUTION_SET = [
#     (1024, 1024),
#     (1152, 896),
#     (1216, 832),
#     (1344, 768),
#     (1536, 640),
# ]
RESOLUTION_CONFIG = {
    # 1328: [
    #     (1328, 1328), # 1:1
    #     (928, 1664), # 9:16
    #     (1140, 1472), # 3:4
    # ],
    1328: [
        (1328, 1328),
        (880, 2032),
        (912, 1952),
        (960, 1904),
        (992, 1824),
        (1040, 1744),
        (1072, 1616),
        (1168, 1536),
        (1248, 1456),
    ],
    1024: [
        (1024, 1024),
        (672, 1568),
        (704, 1504),
        (736, 1472),
        (768, 1408),
        (800, 1344),
        (832, 1248),
        (896, 1184),
        (960, 1120),
    ],
    768: [
        (768, 768),
        (512, 1184),
        (512, 1152),
        (544, 1088),
        (576, 1056),
        (608, 992),
        (640, 960),
        (672, 896),
        (704, 832),
    ],
    # based on 1024 to create 512
    512: [
        (512, 512),
        (352, 800),
        (352, 768),
        (384, 736),
        (384, 704),
        (416, 672),
        (416, 640),
        (448, 608),
        (480, 576),
    ],
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
    height, width, _ = image.shape
    resolution_set = RESOLUTION_CONFIG[resolution]
    
    # get ratio
    image_ratio = width / height

    target_set = resolution_set.copy()
    reversed_set = [(y, x) for x, y in target_set]
    target_set = sorted(set(target_set + reversed_set))
    target_ratio = list(set([round(width/height, 2) for width,height in target_set]))
    
    # Find the closest vertical ratio
    closest_ratio = min(target_ratio, key=lambda x: abs(x - image_ratio))
    closest_resolution = target_set[target_ratio.index(closest_ratio)]

    return closest_ratio,closest_resolution

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

@torch.no_grad()
def vae_encode(vae,image):
    # create tensor latent
    
    pixel_values = []
    pixel_values.append(image)
    pixel_values = torch.stack(pixel_values).to(vae.device)
    # del image
    
    # Qwen expects a `num_frames` dimension too.
    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(2)
        
    with torch.no_grad():
        latent = vae.encode(pixel_values).latent_dist.sample().squeeze(0)
        
        del pixel_values
    latent_dict = {
        'latent': latent.cpu()
    }
    return latent_dict

def read_image(image_path):
    try:
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is not None:
            # Convert to RGB format (assuming the original image is in BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            print(f"Failed to open {image_path}.")
    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")
    return image


def crop_image(image_path,resolution):
    image = read_image(image_path)
    ##############################################################################
    # Simple center crop for others
    ##############################################################################
    # width, height = image.size
    # original_size = (height, width)
    # image = numpy.array(image)
    
    height, width, _ = image.shape
    # original_size = (height, width)
    
    # get nearest resolution
    closest_ratio,closest_resolution = get_nearest_resolution(image,resolution=resolution)
    # we need to expand the closest resolution to target resolution before cropping
    scale_ratio = closest_resolution[0] / closest_resolution[1]
    image_ratio = width / height
    scale_with_height = True
    # crops_coords_top_left = (0,0)
    # referenced kohya ss code
    if image_ratio < scale_ratio: 
        scale_with_height = False
    try:
        # image = simple_center_crop(image,scale_with_height,closest_resolution)
        image,crop_x,crop_y = simple_center_crop(image,scale_with_height,closest_resolution)
        # crops_coords_top_left = (crop_y,crop_x)
        # save_webp(simple_crop_image,filename,'simple',os.path.join(output_dir,"simple"))
    except Exception as e:
        print(e)
        raise e
    # test = Image.fromarray(image)
    # test.show()
    # set meta data
    return image

def compute_text_embeddings(text_encoders, tokenizers, prompt, device, image=None, processor=None):
    with torch.no_grad():
        prompt_embeds, prompt_embeds_mask = encode_prompt(text_encoders, tokenizers, prompt, device=device, image=image, processor=processor)
        prompt_embeds = prompt_embeds.to(device)
        # text_ids = text_ids.to(device)
    return prompt_embeds, prompt_embeds_mask #, text_ids


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

    prompt_embeds, prompt_embeds_mask = encode_prompt(text_encoders,tokenizers,"")
    prompt_embed = prompt_embeds.squeeze(0)
    prompt_embeds_mask = prompt_embeds_mask.squeeze(0)
    
    latent = {
        "prompt_embed": prompt_embed.cpu(), 
        "prompt_embeds_mask": prompt_embeds_mask.cpu(),
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
    processor=None
):
    device = text_encoder.device
    dtype = text_encoder.dtype
    
    prompt = [prompt] if isinstance(prompt, str) else prompt

    if image is None and processor is None:
        template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 34
        txt = [template.format(e) for e in prompt]
        
        model_inputs = tokenizer(
            txt, max_length=max_sequence_length + drop_idx, padding=True, truncation=True, return_tensors="pt"
        ).to(device) 
        
        outputs = text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            output_hidden_states=True,
        )
    else:
        template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 64
        txt = [template.format(e) for e in prompt]
        
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
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )
       
    hidden_states = outputs.hidden_states[-1]
    split_hidden_states = extract_masked_hidden(hidden_states, model_inputs.attention_mask)
    split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
    attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
    max_seq_len = max([e.size(0) for e in split_hidden_states])
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
    )
    encoder_attention_mask = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
    )

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    return prompt_embeds, encoder_attention_mask

def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length=1024,
    device=None,
    image=None,
    processor=None
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds, prompt_embeds_mask = encode_prompt_with_qwenvl(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        device=device if device is not None else text_encoders[-1].device,
        image=image,
        processor=processor
    )
    return prompt_embeds, prompt_embeds_mask
    
def simple_center_crop(image,scale_with_height,closest_resolution):
    height, width, _ = image.shape
    # print("ori size:",width,height)
    if scale_with_height: 
        up_scale = height / closest_resolution[1]
    else:
        up_scale = width / closest_resolution[0]

    expanded_closest_size = (int(closest_resolution[0] * up_scale + 0.5), int(closest_resolution[1] * up_scale + 0.5))
    
    diff_x = abs(expanded_closest_size[0] - width)
    diff_y = abs(expanded_closest_size[1] - height)

    crop_x = 0
    crop_y = 0
    # crop extra part of the resized images
    if diff_x>0:
        crop_x =  diff_x //2
        cropped_image = image[:,  crop_x:width-diff_x+crop_x]
    elif diff_y>0:
        crop_y =  diff_y//2
        cropped_image = image[crop_y:height-diff_y+crop_y, :]
    else:
        # 1:1 ratio
        cropped_image = image

    # print(f"ori ratio:{width/height}")
    height, width, _ = cropped_image.shape  
    # print(f"cropped ratio:{width/height}")
    # print(f"closest ratio:{closest_resolution[0]/closest_resolution[1]}")
    # resize image to target resolution
    # return cv2.resize(cropped_image, closest_resolution)
    # return resize(cropped_image,closest_resolution, resize_method="fs_resize"),crop_x,crop_y
    return resize(cropped_image, closest_resolution, resize_method="lanczos"), crop_x, crop_y

