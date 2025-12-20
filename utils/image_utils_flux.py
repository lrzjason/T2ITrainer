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

from utils.utils import ToTensorUniversal, vae_encode_utils

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
    target_ratio = sorted(set([round(width/height, 2) for width,height in target_set]))
    
    # Find the closest vertical ratio
    closest_ratio = min(target_ratio, key=lambda x: abs(x - image_ratio))
    closest_resolution = target_set[target_ratio.index(closest_ratio)]

    return closest_ratio,closest_resolution


##input: datarows -> output: metadata
#looks like leftover code from leftover_idx, check, then delete
class CachedImageDataset(Dataset):
    def __init__(self, datarows,conditional_dropout_percent=0.1): 
        self.datarows = datarows
        self.leftover_indices = []  #initialize an empty list to store indices of leftover items
        #for conditional_dropout
        self.conditional_dropout_percent = conditional_dropout_percent
        embedding = get_empty_embedding()
        self.empty_prompt_embed = embedding['prompt_embed']  # Tuple of (empty_prompt_embed, empty_pooled_prompt_embed)
        self.empty_pooled_prompt_embed = embedding['pooled_prompt_embed']
        self.empty_txt_attention_mask = embedding['txt_attention_mask']
        # self.empty_text_id = embedding['text_id']
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

        #cached files
        cached_npz = torch.load(metadata['npz_path'], weights_only=True)
        cached_latent = torch.load(metadata['latent_path'], weights_only=True)
        latent = cached_latent['latent']
        prompt_embed = cached_npz['prompt_embed']
        pooled_prompt_embed = cached_npz['pooled_prompt_embed']
        txt_attention_mask = cached_npz['txt_attention_mask']
        # text_id = cached_npz['text_id']

        # conditional_dropout
        if random.random() < self.conditional_dropout_percent:
            prompt_embed = self.empty_prompt_embed
            pooled_prompt_embed = self.empty_pooled_prompt_embed
            txt_attention_mask = self.empty_txt_attention_mask
            # text_id = self.empty_text_id

        result = {
            "latent": latent,
            "prompt_embed": prompt_embed,
            "pooled_prompt_embed": pooled_prompt_embed,
            "txt_attention_mask": txt_attention_mask,
            # "text_id": text_id,
        }
        
        if "redux_prompt_embed" in cached_npz:
            result["redux_prompt_embed"] = cached_npz['redux_prompt_embed']
            result["redux_pooled_prompt_embed"] = cached_npz['redux_pooled_prompt_embed']
            
        return result
    
class CachedMutiImageDatasetKontext(Dataset):
    def __init__(self, datarows,conditional_dropout_percent=0.1, has_redux=False, dataset_configs=None): 
        self.has_redux = has_redux
        self.datarows = datarows
        self.leftover_indices = []  #initialize an empty list to store indices of leftover items
        #for conditional_dropout
        self.conditional_dropout_percent = conditional_dropout_percent
        self.dataset_configs = dataset_configs
        self.empty_embedding = get_empty_embedding()
        # self.caption_key = "captions"
        # self.latent_key = "latent"
        # self.latent_path_key = "latent_path"
        # self.extra_keys = [
        #     {
        #         "latent_key":"masked_latent",
        #         "latent_path_key":"masked_latent_path",
        #     }
        # ]
        # self.npz_path_key = "npz_path"
        # self.npz_keys = [
        #     "prompt_embed",
        #     "pooled_prompt_embed",
        #     "txt_attention_mask"
        # ]
        # self.npz_extra_keys = [
        #     "redux_prompt_embed",
        #     "redux_pooled_prompt_embed"
        # ]
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
        npz_extra_keys = self.dataset_configs["npz_extra_keys"]
        latent_path_key = self.dataset_configs["latent_path_key"]
        latent_key = self.dataset_configs["latent_key"]
        # extra_keys = self.dataset_configs["extra_keys"]
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
                    if "npz_extra_keys" in self.dataset_configs:
                        result[caption_key][key]["redux"] = {}
                        for npz_extra_key_group in npz_extra_keys.keys():
                            result[caption_key][key]["redux"][npz_extra_key_group] = {}
                            for npz_extra_key in npz_extra_keys[npz_extra_key_group]:
                                result[caption_key][key]["redux"][npz_extra_key_group][npz_extra_key] = cached_npz["redux"][npz_extra_key_group][npz_extra_key]
                        
            if latent_path_key in metadata[key]:
                latent = torch.load(metadata[key][latent_path_key], weights_only=True)
                # for captions
                if key in result:
                    result[key][latent_key] = latent[latent_key]
                else:
                    result[key] = {
                        latent_key:latent[latent_key]
                    }
                # if "extra_keys" in self.dataset_configs:
                #     for extra_key_group in extra_keys.keys():
                #         extra_key = extra_keys[extra_key_group]
                #         extra_latent_key = extra_key["latent_key"]
                #         extra_latent_path_key = extra_key["latent_path_key"]
                #         if extra_latent_path_key in metadata[key]:
                #             extra_latent = torch.load(metadata[key][extra_latent_path_key], weights_only=True)
                #             result[key][extra_latent_key] = extra_latent[latent_key]
        return result
    
class CachedMutiImageDataset(Dataset):
    def __init__(self, datarows,conditional_dropout_percent=0.1, has_redux=False, dataset_configs=None): 
        self.has_redux = has_redux
        self.datarows = datarows
        self.leftover_indices = []  #initialize an empty list to store indices of leftover items
        #for conditional_dropout
        self.conditional_dropout_percent = conditional_dropout_percent
        self.dataset_configs = dataset_configs
        self.empty_embedding = get_empty_embedding()
        # self.caption_key = "captions"
        # self.latent_key = "latent"
        # self.latent_path_key = "latent_path"
        # self.extra_keys = [
        #     {
        #         "latent_key":"masked_latent",
        #         "latent_path_key":"masked_latent_path",
        #     }
        # ]
        # self.npz_path_key = "npz_path"
        # self.npz_keys = [
        #     "prompt_embed",
        #     "pooled_prompt_embed",
        #     "txt_attention_mask"
        # ]
        # self.npz_extra_keys = [
        #     "redux_prompt_embed",
        #     "redux_pooled_prompt_embed"
        # ]
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
        # {
        #    "captions":{
        #       "factual":{
        #          "text_path":"F:/ImageSet/ObjectRemoval/new_construct\\title_removal\\cat_2_F.txt",
        #          "text_path_md5":"8522dcd4f1a2b6c4aff94efdc41579c3",
        #          "npz_path":"F:/ImageSet/ObjectRemoval/new_construct\\title_removal\\cat_2_F.npflux"
        #       }
        #    },
        #    "mapping_key":"F:/ImageSet/ObjectRemoval/new_construct\\title_removal_cat_2_",
        #    "factual":{
        #       "image_path":"F:/ImageSet/ObjectRemoval/new_construct\\title_removal\\cat_2_F.webp",
        #       "latent_path":"F:/ImageSet/ObjectRemoval/new_construct\\title_removal\\cat_2_F.npfluxlatent",
        #       "latent_path_md5":"cf8043c622b1d5fe1b0ba155bd0cd4de"
        #    },
        #    "groundtrue":{
        #       "image_path":"F:/ImageSet/ObjectRemoval/new_construct\\title_removal\\cat_2_G.webp",
        #       "latent_path":"F:/ImageSet/ObjectRemoval/new_construct\\title_removal\\cat_2_G.npfluxlatent",
        #       "latent_path_md5":"0942d20edd4a4fb2f7d05c91958aaca0"
        #    },
        #    "factual_mask":{
        #       "image_path":"F:/ImageSet/ObjectRemoval/new_construct\\title_removal\\cat_2_M.png",
        #       "latent_path":"F:/ImageSet/ObjectRemoval/new_construct\\title_removal\\cat_2_M.npfluxlatent",
        #       "masked_latent_path":"F:/ImageSet/ObjectRemoval/new_construct\\title_removal\\cat_2_M_masked.npfluxlatent"
        #    },
        #    "bucket":"448x576"
        # }
        
        
        # dataset_configs = {
        #     "caption_key":caption_key,
        #     "latent_key":"latent",
        #     "latent_path_key":latent_path_key,
        #     "extra_keys":{
        #         image_1_mask=factual_mask:{
        #             "latent_key":f"{masked_suffix}_{latent_key}",
        #             "latent_path_key":f"{masked_suffix}_{latent_path_key}",
        #         }
        #     },
        #     "npz_path_key": embbeding_path_key,
        #     "npz_keys": {
        #         prompt_embed_key:prompt_embed_key,
        #         pool_prompt_embed_key:pool_prompt_embed_key,
        #         txt_attention_mask_key:txt_attention_mask_key
        #     },
        #     "npz_extra_keys": {
        #         redux_key:[
        #             prompt_embed_key,
        #             pool_prompt_embed_key
        #         ]
        #     }
        # }
        result = {
            
        }
        metadata_caption_key = self.dataset_configs["caption_key"]
        npz_path_key = self.dataset_configs["npz_path_key"]
        npz_keys = self.dataset_configs["npz_keys"]
        npz_extra_keys = None
        if "npz_extra_keys" in self.dataset_configs:
            npz_extra_keys = self.dataset_configs["npz_extra_keys"]
        latent_path_key = self.dataset_configs["latent_path_key"]
        latent_key = self.dataset_configs["latent_key"]
        extra_keys = None 
        if "extra_keys" in self.dataset_configs:
            extra_keys = self.dataset_configs["extra_keys"]
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
                    if "npz_extra_keys" in self.dataset_configs:
                        result[caption_key][key]["redux"] = {}
                        for npz_extra_key_group in npz_extra_keys.keys():
                            result[caption_key][key]["redux"][npz_extra_key_group] = {}
                            for npz_extra_key in npz_extra_keys[npz_extra_key_group]:
                                result[caption_key][key]["redux"][npz_extra_key_group][npz_extra_key] = cached_npz["redux"][npz_extra_key_group][npz_extra_key]
                        
            if latent_path_key in metadata[key]:
                latent = torch.load(metadata[key][latent_path_key], weights_only=True)
                # for captions
                if key in result:
                    result[key][latent_key] = latent[latent_key]
                else:
                    result[key] = {
                        latent_key:latent[latent_key]
                    }
                if "extra_keys" in self.dataset_configs:
                    for extra_key_group in extra_keys.keys():
                        extra_key = extra_keys[extra_key_group]
                        extra_latent_key = extra_key["latent_key"]
                        extra_latent_path_key = extra_key["latent_path_key"]
                        if extra_latent_path_key in metadata[key]:
                            extra_latent = torch.load(metadata[key][extra_latent_path_key], weights_only=True)
                            result[key][extra_latent_key] = extra_latent[latent_key]
        return result

class CachedMaskedPairsDataset(Dataset):
    def __init__(self, datarows,conditional_dropout_percent=0.1, has_redux=False): 
        self.has_redux = has_redux
        self.datarows = datarows
        self.leftover_indices = []  #initialize an empty list to store indices of leftover items
        #for conditional_dropout
        self.conditional_dropout_percent = conditional_dropout_percent
        # embedding = get_empty_embedding()
        # self.empty_prompt_embed = embedding['prompt_embed']  # Tuple of (empty_prompt_embed, empty_pooled_prompt_embed)
        # self.empty_pooled_prompt_embed = embedding['pooled_prompt_embed']
        # self.empty_txt_attention_mask = embedding['txt_attention_mask']
        # self.empty_text_id = embedding['text_id']
        
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

        #cached files
        cached_npz = torch.load(metadata['npz_path'], weights_only=True)
        prompt_embed = cached_npz['prompt_embed']
        pooled_prompt_embed = cached_npz['pooled_prompt_embed']
        txt_attention_mask = cached_npz['txt_attention_mask']
        
        result = {
            "prompt_embed": prompt_embed,
            "pooled_prompt_embed": pooled_prompt_embed,
            "txt_attention_mask": txt_attention_mask,
            # "text_id": text_id,
        }
        if "redux_prompt_embed" in cached_npz:
            result["redux_prompt_embed"] = cached_npz['redux_prompt_embed']
            result["redux_pooled_prompt_embed"] = cached_npz['redux_pooled_prompt_embed']
        
        cached_latent_names = ["ground_true", "factual_image", "factual_image_mask", "factual_image_masked_image"]
        if self.has_redux:
            cached_latent_names = ["ground_true", "factual_image", "factual_image_mask", "factual_image_masked_image", "redux_image"]
            
        for cached_latent_name in cached_latent_names:
            if not "latent_path" in metadata[cached_latent_name]:
                raise ValueError(f"{cached_latent_name} is not in metadata")
            cached_latent = torch.load(metadata[cached_latent_name]['latent_path'], weights_only=True)
            result[cached_latent_name] = cached_latent
        return result

# main idea is store all tensor related in .npz file
# other information stored in .json
@torch.no_grad()
def create_metadata_cache(tokenizers,text_encoders,vae,image_files,recreate_cache=False, metadata_path="metadata_sd35.json", resolution_config="1024",pipe_prior_redux=None):
    create_empty_embedding(tokenizers,text_encoders)
    datarows = []
    embedding_objects = []
    resolutions = resolution_config.split(",")
    resolutions = [int(resolution) for resolution in resolutions]
    for image_file in tqdm(image_files):
        file_name = os.path.basename(image_file)
        folder_path = os.path.dirname(image_file)
        
        # for resolution in resolutions:
        json_obj = create_embedding(
            tokenizers,text_encoders,folder_path,file_name,
            resolutions=resolutions,recreate_cache=recreate_cache,
            pipe_prior_redux=pipe_prior_redux)
        
        embedding_objects.append(json_obj)
    
    # move glm to cpu to reduce vram memory
    # text_encoders[0].to("cpu")
    del text_encoders,tokenizers
    flush()
    # cache latent
    print("Cache latent")
    for json_obj in tqdm(embedding_objects):
        for resolution in resolutions:
            full_obj = cache_file(vae,json_obj,resolution=resolution,recreate_cache=recreate_cache)
            datarows.append(full_obj)
    # Serializing json
    json_object = json.dumps(datarows, indent=4)
    
    # Writing to metadata.json
    with open(metadata_path, "w", encoding='utf-8') as outfile:
        outfile.write(json_object)
    
    return datarows

@torch.no_grad()
def create_embedding(tokenizers,text_encoders,folder_path,file,cache_ext=".npflux",
                    resolutions=None,recreate_cache=False,pipe_prior_redux=None,
                    exist_npz_path="",redux_image_path=""):
    # get filename and ext from file
    filename, _ = os.path.splitext(file)
    image_path = os.path.join(folder_path, file)
    image_path_md5 = get_md5_by_path(image_path)
    if resolutions is None:
        resolutions = [1024]
    json_obj = {
        'image_path':image_path,
        'image_path_md5':image_path_md5,
        'folder_path':folder_path,
        'file':file,
        'resolutions':resolutions
    }
    # fix init prompt
    # json_obj['prompt'] = ''
    # read caption
    caption_ext = '.txt'
    text_path = os.path.join(folder_path, f'{filename}{caption_ext}')
    content = ''
    if os.path.exists(text_path):
        json_obj["text_path"] = text_path
        try:
            content = open(text_path, encoding='utf-8').read()
            # json_obj['prompt'] = content
            json_obj["text_path_md5"] = get_md5_by_path(text_path)
        except:
            content = open(text_path, encoding='utf-8').read()
            # try to remove non utf8 character
            content = replace_non_utf8_characters(content)
            # json_obj['prompt'] = content
            json_obj["text_path_md5"] = ""
            

    file_path = os.path.join(folder_path, filename)
    npz_path = f'{file_path}{cache_ext}'
    if exist_npz_path != "" and os.path.exists(exist_npz_path):
        npz_path = exist_npz_path
    json_obj["npz_path"] = npz_path
    
    if not recreate_cache and os.path.exists(npz_path):
        if 'npz_path_md5' not in json_obj:
            json_obj["npz_path_md5"] = get_md5_by_path(npz_path)
        return json_obj
    
    prompt_embeds, pooled_prompt_embeds, txt_attention_masks = compute_text_embeddings(text_encoders,tokenizers,content,device=text_encoders[0].device)
    prompt_embed = prompt_embeds.squeeze(0)
    pooled_prompt_embed = pooled_prompt_embeds.squeeze(0)
    txt_attention_mask = txt_attention_masks.squeeze(0)
    # text_id = text_ids.squeeze(0)
    
    # try:
    #     image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    #     if image is not None:
    #         # Convert to RGB format (assuming the original image is in BGR)
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     else:
    #         print(f"Failed to open {image_path}.")
    # except Exception as e:
    #     print(f"An error occurred while processing {image_path}: {e}")

    # height, width, _ = image.shape
    # original_size = (height, width)
    # crops_coords_top_left = (0,0)
    # time_id = torch.tensor(list(original_size + crops_coords_top_left + original_size)).to(text_encoders[0].device, dtype=text_encoders[0].dtype)
    
    npz_dict = {
        "prompt_embed": prompt_embed.cpu(), 
        "pooled_prompt_embed": pooled_prompt_embed.cpu(),
        "txt_attention_mask": txt_attention_mask.cpu(),
        # "text_id": text_ids.cpu(),
        # "time_id": time_id.cpu()
    }
    
    if pipe_prior_redux is not None:
        if redux_image_path !="":
            image_path = redux_image_path
        image = load_image(image_path)
        pipe_prior_output = pipe_prior_redux(image,prompt_embeds=prompt_embeds,pooled_prompt_embeds=pooled_prompt_embeds)
        prompt_embed = pipe_prior_output.prompt_embeds.squeeze(0)
        pooled_prompt_embed = pipe_prior_output.pooled_prompt_embeds.squeeze(0)
        npz_dict["redux_prompt_embed"] = prompt_embed.cpu()
        npz_dict["redux_pooled_prompt_embed"] = pooled_prompt_embed.cpu()
        
        # extend txt_attention_mask dim as 1 to match prompt_embed
        # npz_dict["txt_attention_mask"] = torch.ones_like(prompt_embed)
    
    # save latent to cache file
    torch.save(npz_dict, npz_path)
    return json_obj

# based on image_path, caption_path, caption create json object
# write tensor related to npz file
@torch.no_grad()
def cache_file(vae,json_obj,resolution=1024,cache_ext=".npflux",latent_ext=".npfluxlatent",recreate_cache=False):
    npz_path = json_obj["npz_path"]
    
    
    latent_cache_path = npz_path.replace(cache_ext,latent_ext)
    if resolution > 1024:
        latent_cache_path = npz_path.replace(cache_ext,f"_{resolution}{latent_ext}")
    json_obj["latent_path"] = latent_cache_path
    
    
    npz_dict = {}
    if os.path.exists(npz_path):
        try:
            npz_dict = torch.load(npz_path, weights_only=True)
        except:
            print(f"Failed to load {npz_path}")
    image_path = json_obj["image_path"]
    # resolution = json_obj["resolution"]
    
    try:
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is not None:
            # Convert to RGB format (assuming the original image is in BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            print(f"Failed to open {image_path}.")
    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")

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
    image_height, image_width, _ = image.shape
    # target_size = (image_height,image_width)
    ##############################################################################
    
    json_obj['bucket'] = f"{image_width}x{image_height}"
    
    # time_id = torch.tensor(list(original_size + crops_coords_top_left + target_size)).to(vae.device, dtype=vae.dtype)

    # skip if already cached
    if os.path.exists(latent_cache_path) and not recreate_cache:
        if 'latent_path_md5' not in json_obj:
            json_obj['latent_path_md5'] = get_md5_by_path(latent_cache_path)
            json_obj['npz_path_md5'] = get_md5_by_path(npz_path)
        return json_obj
    
    train_transforms = transforms.Compose([ToTensorUniversal, transforms.Normalize([0.5], [0.5])])
    image = train_transforms(image)
    
    # create tensor latent
    pixel_values = []
    pixel_values.append(image)
    pixel_values = torch.stack(pixel_values).to(vae.device)
    del image
    
    with torch.no_grad():
        #contiguous_format = (contiguous memory block), unsqueeze(0) adds bsz 1 dimension, else error: but got weight of shape [128] and input of shape [128, 512, 512]
        latent = vae.encode(pixel_values).latent_dist.sample().squeeze(0)
        # .squeeze(0) #squeeze to remove bsz dimension
        # latent = latent * vae.config.scaling_factor
        
        del pixel_values
        # print(latent.shape) torch.Size([4, 144, 112])

    latent_dict = {
        'latent': latent.cpu()
    }
    torch.save(latent_dict, latent_cache_path)
    # latent_dict['latent'] = latent.cpu()
    # npz_dict['time_id'] = time_id.cpu()
    npz_dict['latent_path'] = latent_cache_path
    json_obj['latent_path_md5'] = get_md5_by_path(latent_cache_path)
    # save latent to cache file
    torch.save(npz_dict, npz_path)
    json_obj['npz_path_md5'] = get_md5_by_path(npz_path)
    del npz_dict
    flush()
    return json_obj



@torch.no_grad()
def vae_encode(vae,image):
    # create tensor latent
    latent_dict = vae_encode_utils(vae,image, vae_type="flux")   
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

# based on image_path, caption_path, caption create json object
# write tensor related to npz file
# ground_true_file, factual_image_file and factual_image_mask
@torch.no_grad()
def cache_multiple(vae,json_obj,resolution=1024,cache_ext=".npflux",latent_ext=".npfluxlatent",recreate_cache=False,use_original_mask=False):
    npz_path = json_obj["npz_path"]
    
    # npz_dict = {}
    # if os.path.exists(npz_path):
    #     try:
    #         npz_dict = torch.load(npz_path, weights_only=True)
    #     except:
    #         print(f"Failed to load {npz_path}")
    
    image_files = [ 
        ("factual_image", json_obj["factual_image_path"]),
        ("ground_true", json_obj["ground_true_path"])
        # ("factual_image_mask", json_obj["factual_image_mask_path"])
    ]
    # factual_image_file = json_obj["factual_image_file"]
    # factual_image_mask = json_obj["factual_image_mask"]
    
    # image_path = json_obj["image_path"]
    # resolution = json_obj["resolution"]
    factual_image = None
    f_height = 0
    f_width = 0
    
    train_transforms = transforms.Compose([ToTensorUniversal(), transforms.Normalize([0.5], [0.5])])
    
    for image_class,image_path in image_files:
        filename, _ = os.path.splitext(image_path)
        
        # create image class if not exist
        if not image_class in json_obj:
            json_obj[image_class] = {}
        json_obj[image_class]["latent_path"] = f"{filename}{latent_ext}"
        latent_cache_path = f"{filename}{latent_ext}"
        # latent_cache_path = npz_path.replace(cache_ext,latent_ext)
        # json_obj["latent_path"] = latent_cache_path
        
        
        # target_size = (image_height,image_width)
        ##############################################################################
        
        image = crop_image(image_path,resolution)
        image_height, image_width, _ = image.shape
        
        # if cache_ratio !=1:
        #     # resize image with cache_ratio
        #     image = cv2.resize(image, (int(image_width*cache_ratio), int(image_height*cache_ratio)), interpolation=cv2.INTER_LINEAR)
        #     image_height, image_width, _ = image.shape
        
        json_obj['bucket'] = f"{image_width}x{image_height}"
        
        # time_id = torch.tensor(list(original_size + crops_coords_top_left + target_size)).to(vae.device, dtype=vae.dtype)

        # skip if already cached
        if os.path.exists(latent_cache_path) and not recreate_cache:
            if 'latent_path_md5' not in json_obj[image_class]:
                json_obj[image_class]['latent_path_md5'] = get_md5_by_path(latent_cache_path)
                json_obj[image_class]['npz_path_md5'] = get_md5_by_path(npz_path)
            continue
        
        image = train_transforms(image)
        if image_class == "factual_image":
            factual_image = image.unsqueeze(0)
            f_height = image_height
            f_width = image_width
            
        latent_dict = vae_encode(vae,image)
        torch.save(latent_dict, latent_cache_path)
        json_obj[image_class]['latent_path_md5'] = get_md5_by_path(latent_cache_path)
    # del npz_dict
    flush()
    
    if "factual_image_masked_image" in json_obj and "factual_image_mask" in json_obj and not recreate_cache:
        return json_obj
    
    # prepare mask latent
    vae_scale_factor = (
        2 ** (len(vae.config.block_out_channels) - 1)
    )
    
    if "redux_image_path" in json_obj and "redux_image" not in json_obj:
        redux_image_path = json_obj["redux_image_path"]
        filename, _ = os.path.splitext(redux_image_path)
        redux_cache_path = f"{filename}{latent_ext}"
        
        redux_image = read_image(redux_image_path)
        # if cache_ratio !=1:
            # resize image with cache_ratio
        redux_image = cv2.resize(redux_image, (int(f_width), int(f_height)), interpolation=cv2.INTER_LANCZOS4)
            
        redux_image = train_transforms(redux_image)
        latent_dict = vae_encode(vae,redux_image)
        torch.save(latent_dict, redux_cache_path)
        json_obj["redux_image"] = {}
        json_obj["redux_image"]["latent_path"] = redux_cache_path
        json_obj["redux_image"]['latent_path_md5'] = get_md5_by_path(redux_cache_path)
    else:
        json_obj["redux_image"] = {}
        json_obj["redux_image"]["latent_path"] = ""
        
    mask_path = json_obj["factual_image_mask_path"]
    filename, _ = os.path.splitext(mask_path)
    masked_image_latent_cache_path = f"{filename}_masked_image{latent_ext}"
    mask_cache_path = f"{filename}{latent_ext}"
    if "factual_image_masked_image" not in json_obj:
        json_obj["factual_image_masked_image"] = {}
    json_obj["factual_image_masked_image"]["latent_path"] = masked_image_latent_cache_path
    if "factual_image_mask" not in json_obj:
        json_obj["factual_image_mask"] = {}
    json_obj["factual_image_mask"]["latent_path"] = mask_cache_path
    if os.path.exists(masked_image_latent_cache_path) and os.path.exists(mask_cache_path) and not recreate_cache:
        return json_obj
    
    mask_processor = VaeImageProcessor(
        vae_scale_factor=vae_scale_factor * 2,
        vae_latent_channels=vae.config.latent_channels,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True,
        do_resize=False
    )
    
    mask_image = crop_image(mask_path,resolution)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGB2GRAY)
    
    # if cache_ratio !=1:
    #     # resize image with cache_ratio
    #     mask_image = cv2.resize(mask_image, (int(f_width), int(f_height)), interpolation=cv2.INTER_LINEAR)
    
    mask_image = mask_processor.preprocess(mask_image, height=f_height, width=f_width)

    masked_image = factual_image * (1 - mask_image)
    masked_image = masked_image.to(device=vae.device)

    height, width = factual_image.shape[-2:]
    
    
    # 1. calculate the height and width of the latents
    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    # 2. encode the masked image
    masked_image_latent = vae.encode(masked_image).latent_dist.sample().squeeze(0)

    # for debug, return the original mask
    
    if not use_original_mask:
        # 5.resize mask to latents shape we we concatenate the mask to the latents
        mask_image = mask_image[:, 0, :, :]  # batch_size, 8 * height, 8 * width (mask has not been 8x compressed)
        mask_image = mask_image.view(
            1, height, vae_scale_factor, width, vae_scale_factor
        )  # batch_size, height, 8, width, 8
        mask_image = mask_image.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
        mask_image = mask_image.reshape(
            1, vae_scale_factor * vae_scale_factor, height, width
        )  # batch_size, 8*8, height, width

    latent_dict = {
        'latent': masked_image_latent.cpu()
    }
    # store masked image
    if not "factual_image_masked_image" in json_obj:
        json_obj["factual_image_masked_image"] = {}
    latent_cache_path = f"{filename}_masked_image{latent_ext}"
    json_obj["factual_image_masked_image"]["latent_path"] = latent_cache_path
    torch.save(latent_dict, latent_cache_path)
    json_obj["factual_image_masked_image"]['latent_path_md5'] = get_md5_by_path(latent_cache_path)
    
    if not "factual_image_mask" in json_obj:
        json_obj["factual_image_mask"] = {}
    latent_cache_path = f"{filename}{latent_ext}"
    json_obj["factual_image_mask"]["latent_path"] = latent_cache_path
    latent_dict = {
        'latent': mask_image.squeeze().cpu()
    }
    torch.save(latent_dict, latent_cache_path)
    json_obj["factual_image_mask"]['latent_path_md5'] = get_md5_by_path(latent_cache_path)
    
    
    
    return json_obj

def compute_text_embeddings(text_encoders, tokenizers, prompt, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, txt_attention_masks = encode_prompt(text_encoders, tokenizers, prompt, device=device)
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        # text_ids = text_ids.to(device)
    return prompt_embeds, pooled_prompt_embeds, txt_attention_masks #, text_ids


def get_empty_embedding(cache_path="cache/empty_embedding.npflux"):
    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=True)
    else:
        raise FileNotFoundError(f"{cache_path} not found")
def create_empty_embedding(tokenizers,text_encoders,cache_path="cache/empty_embedding.npflux",recreate=False, resolution=1024):
    if recreate:
        os.remove(cache_path)

    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=True)

    prompt_embeds, pooled_prompt_embeds, txt_attention_masks = encode_prompt(text_encoders,tokenizers,"")
    prompt_embed = prompt_embeds.squeeze(0)
    pooled_prompt_embed = pooled_prompt_embeds.squeeze(0)
    txt_attention_mask = txt_attention_masks.squeeze(0)
    # text_id = text_ids.squeeze(0)
    # time_id = torch.tensor([
    #     # original size
    #     resolution,resolution,
    #     0,0,
    #     # target size
    #     resolution,resolution
    # ])
    latent = {
        "prompt_embed": prompt_embed.cpu(), 
        "pooled_prompt_embed": pooled_prompt_embed.cpu(),
        "txt_attention_mask": txt_attention_mask.cpu(),
        # "text_id": text_id.cpu(),
        # "time_id":time_id.cpu()
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


def encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        txt_attention_mask = text_inputs.attention_mask
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, txt_attention_mask


def encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    # pooled_prompt_embeds = prompt_embeds[0]
    # prompt_embeds = prompt_embeds.hidden_states[-2]
    # prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # _, seq_len, _ = prompt_embeds.shape
    # # duplicate text embeddings for each generation per prompt, using mps friendly method
    # prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    # prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds #, pooled_prompt_embeds

# def encode_prompt_with_glm(
#     text_encoder,
#     tokenizer,
#     prompt: str,
#     device=None,
#     num_images_per_prompt: int = 1,
# ):
#     prompt = [prompt] if isinstance(prompt, str) else prompt
#     # batch_size = len(prompt)

#     text_inputs = tokenizer(
#         prompt,
#         padding="max_length",
#         max_length=256,
#         truncation=True,
#         return_tensors="pt",
#     ).to(device)

#     output = text_encoder(
#             input_ids=text_inputs['input_ids'],
#             attention_mask=text_inputs['attention_mask'],
#             position_ids=text_inputs['position_ids'],
#             output_hidden_states=True)
#     # text_input_ids = text_inputs.input_ids
#     # prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
#     prompt_embeds = output.hidden_states[-2].permute(1, 0, 2).clone()
#     pooled_prompt_embeds = output.hidden_states[-1][-1, :, :].clone() # [batch_size, 4096]
#     bs_embed, seq_len, _ = prompt_embeds.shape
#     prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
#     prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
    
#     pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
#         bs_embed * num_images_per_prompt, -1
#     )
    
#     return prompt_embeds, pooled_prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length=512,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    dtype = text_encoders[0].dtype

    clip_tokenizer = tokenizers[0]
    clip_text_encoder = text_encoders[0]

    # clip_prompt_embeds_list = []
    # clip_pooled_prompt_embeds_list = []
    # for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
    pooled_prompt_embeds = encode_prompt_with_clip(
        text_encoder=clip_text_encoder,
        tokenizer=clip_tokenizer,
        prompt=prompt,
        device=device if device is not None else clip_text_encoder.device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )
    # clip_prompt_embeds_list.append(prompt_embeds)
    # clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    # clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    # pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    prompt_embeds, txt_attention_masks = encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    # clip_prompt_embeds = torch.nn.functional.pad(
    #     clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    # )
    # prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)


    return prompt_embeds, pooled_prompt_embeds, txt_attention_masks
    
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
    return resize(cropped_image,closest_resolution),crop_x,crop_y


# def crop_image(image,resolution=1024):
#     # read image from image_path
    
#     height, width, _ = image.shape
#     # get nearest resolution
#     closest_ratio,closest_resolution = get_nearest_resolution(image,resolution=resolution)
#     # we need to expand the closest resolution to target resolution before cropping
#     scale_ratio = closest_resolution[0] / closest_resolution[1]
#     image_ratio = width / height
#     scale_with_height = True
#     # referenced kohya ss code
#     if image_ratio < scale_ratio: 
#         scale_with_height = False
#     try:
#         image,crop_x,crop_y = simple_center_crop(image,scale_with_height,closest_resolution)
#     except Exception as e:
#         print(e)
#         raise e
    
#     return image,crop_x,crop_y



    