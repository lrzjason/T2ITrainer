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

import numpy as np
from typing import Union

T5_ENCODER = {
    'MT5': 'ckpts/t2i/mt5',
    'attention_mask': True,
    'layer_index': -1,
    'attention_pool': True,
    'torch_dtype': torch.float16,
    'learnable_replace': True
}

BASE_RESOLUTION = 1024

# RESOLUTION_SET = [
#     (1024, 1024),
#     (1152, 896),
#     (1216, 832),
#     (1344, 768),
#     (1536, 640),
# ]

# def get_buckets():
#     horizontal_resolution_set = RESOLUTION_SET
#     vertical_resolution_set = [(height,width) for width,height in RESOLUTION_SET]
#     all_resolution_set = horizontal_resolution_set + vertical_resolution_set[1:]
#     buckets = {}
#     for resolution in all_resolution_set:
#         buckets[f'{resolution[0]}x{resolution[1]}'] = []
#     return buckets

# height / width for pixart
# RESOLUTION_SET = [
#     (1024, 1024), # 1:1
#     (896, 1152),  # 0.7777 3:4 0.75
#     (832, 1216),  # 0.6842 
#     (768, 1344),  # 0.5714 9:16 0.5625
#     (640, 1536),  # 0.4666
# ]


RESOLUTION_SET = [
    (1024, 1024),
    (768, 1280),  # 9:16
    (864, 1152),
    (960, 1280),  # 3:4
]

def get_buckets():
    buckets = {}
    horizontal_resolution_set = RESOLUTION_SET
    vertical_resolution_set = [(width,height) for height,width in RESOLUTION_SET]
    all_resolution_set = horizontal_resolution_set + vertical_resolution_set[1:]
    for resolution in all_resolution_set:
        buckets[f'{resolution[0]}x{resolution[1]}'] = []
    return buckets


# return closest_ratio and width,height closest_resolution
def get_nearest_resolution(image):
    # width, height = image.size

    height, width, _ = image.shape
    
    # get ratio
    # image_ratio = width / height
    image_ratio = height / width

    horizontal_resolution_set = RESOLUTION_SET
    # horizontal_ratio = [round(width/height, 2) for width,height in RESOLUTION_SET]
    horizontal_ratio = [round(height/width, 2) for height,width in RESOLUTION_SET]

    # vertical_resolution_set = [(height,width) for width,height in RESOLUTION_SET]
    # reverse the list as vertical_resolution_set
    vertical_resolution_set = [(width,height) for height,width in RESOLUTION_SET]
    vertical_ratio = [round(height/width, 2) for height,width in vertical_resolution_set]


    target_ratio = horizontal_ratio
    target_set = horizontal_resolution_set
    if width<height:
        target_ratio = vertical_ratio
        target_set = vertical_resolution_set

    # Find the closest vertical ratio
    closest_ratio = min(target_ratio, key=lambda x: abs(x - image_ratio))
    closest_resolution = target_set[target_ratio.index(closest_ratio)]

    return closest_ratio,closest_resolution


#referenced from everyDream discord minienglish1 shared script
#group indices by their corresponding aspect ratio buckets before sampling batches.
class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=True):
        self.dataset = dataset
        self.datarows = dataset.datarows
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.leftover_items = []  #tracks leftover items, without modifying the dataset
        self.bucket_indices = self._bucket_indices_by_aspect_ratio() 

    #groups dataset indices into buckets based on aspect ratio
    def _bucket_indices_by_aspect_ratio(self):
        buckets = {}
        for idx in range(len(self.datarows)): #iterates whole dataset
            closest_bucket_key = self.datarows[idx]['bucket']
            if closest_bucket_key not in buckets: #creates bucket if needed
                buckets[closest_bucket_key] = []
            buckets[closest_bucket_key].append(idx) #adds item to bucket

        for bucket in buckets.values(): #shuffles each bucket's contents
            random.shuffle(bucket)
        return buckets #returns organized buckets

    def __iter__(self): #makes sampler iterable, to be used by PyTorch DataLoader
        #reinitialize bucket_indices - to include leftover items
        self.bucket_indices = self._bucket_indices_by_aspect_ratio()

        #leftover items are distributed to bucket_indices
        if self.leftover_items:
            #same as in def _bucket_indices_by_aspect_ratio(self):
            for leftover_idx in self.leftover_items:
                # closest_bucket = self.dataset[leftover_idx]['bucket']
                closest_bucket_key = self.datarows[leftover_idx]['bucket']
                if closest_bucket_key in self.bucket_indices:
                    self.bucket_indices[closest_bucket_key].insert(0, leftover_idx)
                else:
                    self.bucket_indices[closest_bucket_key] = [leftover_idx]
            self.leftover_items = []  #reset leftover items
        
        all_buckets = list(self.bucket_indices.items())
        random.shuffle(all_buckets)  #shuffle buckets' order, random bucket each batch

        #iterates over buckets, yields when len(batch) == batch size
        for _, bucket_indices in all_buckets: #iterate each bucket
            batch = []
            for idx in bucket_indices: #for a bucket, try to make batch
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch 
                    batch = []
            if not self.drop_last and batch: #if too small
                yield batch  #yield last batch if drop_last is False
            elif batch:  #else store leftovers for the next epoch
                self.leftover_items.extend(batch)  

    def __len__(self):
        #calculates total batches
        total_batches = sum(len(indices) // self.batch_size for indices in self.bucket_indices.values())
        #if using leftovers, append leftovers to total batches
        if not self.drop_last:
            leftovers = sum(len(indices) % self.batch_size for indices in self.bucket_indices.values())
            total_batches += bool(leftovers)  #add one more batch if there are leftovers
        return total_batches
    

##input: datarows -> output: metadata
#looks like leftover code from leftover_idx, check, then delete
class CachedImageDataset(Dataset):
    def __init__(self, datarows,conditional_dropout_percent=0.1): 
        self.datarows = datarows
        self.leftover_indices = []  #initialize an empty list to store indices of leftover items
        #for conditional_dropout
        self.conditional_dropout_percent = conditional_dropout_percent
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

        #cached files
        cached_latent = torch.load(metadata['npz_path'])
        
        latent = cached_latent['latent']
        encoder_hidden_state = cached_latent['encoder_hidden_state']
        text_embedding_mask = cached_latent['text_embedding_mask']
        encoder_hidden_state_t5 = cached_latent['encoder_hidden_state_t5']
        text_embedding_mask_t5 = cached_latent['text_embedding_mask_t5']
        image_meta_size = cached_latent['image_meta_size']
        style = cached_latent['style']
        cos_cis_img = cached_latent['cos_cis_img']
        sin_cis_img = cached_latent['sin_cis_img']
        #conditional_dropout
        if random.random() < self.conditional_dropout_percent:
            encoder_hidden_state = self.empty_embedding['encoder_hidden_state']
            text_embedding_mask = self.empty_embedding['text_embedding_mask']
            encoder_hidden_state_t5 = self.empty_embedding['encoder_hidden_state_t5']
            text_embedding_mask_t5 = self.empty_embedding['text_embedding_mask_t5']

        return {
            "latent":latent,
            "encoder_hidden_state": encoder_hidden_state,
            "text_embedding_mask": text_embedding_mask,
            "encoder_hidden_state_t5": encoder_hidden_state_t5,
            "text_embedding_mask_t5": text_embedding_mask_t5,
            "image_meta_size": image_meta_size,
            "style": style,
            "cos_cis_img": cos_cis_img,
            "sin_cis_img": sin_cis_img,
        }
    
# main idea is store all tensor related in .npz file
# other information stored in .json
def create_metadata_cache(tokenizers,text_encoders,vae,input_dir,caption_exts='.txt,.wd14_cap',recreate=False,recreate_cache=False,  metadata_name="metadata_hy.json"):
    create_empty_embedding(tokenizers,text_encoders)
    supported_image_types = ['.jpg','.jpeg','.png','.webp']
    metadata_path = os.path.join(input_dir, metadata_name)
    if recreate or recreate_cache:
        # remove metadata.json
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
    datarows = []
    # create metadata.jsonl if not exist
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding='utf-8') as readfile:
            datarows = json.loads(readfile.read())
    else:
        # create empty file
        # open(metadata_path, 'w', encoding='utf-8').close()
        for item in tqdm(os.listdir(input_dir),position=0):
            # check item is dir or file
            item_path = os.path.join(input_dir, item)
            # handle subfolders
            if tqdm(os.path.isdir(item_path),position=1):
                folder_path = item_path
                for file in os.listdir(folder_path):
                    for image_type in supported_image_types:
                        if file.endswith(image_type):
                            json_obj = iterate_image(tokenizers,text_encoders,vae,folder_path,file,caption_exts=caption_exts,recreate_cache=recreate_cache)
                            datarows.append(json_obj)
            # handle single files
            else:
                folder_path = input_dir
                file = item
                for image_type in supported_image_types:
                    if file.endswith(image_type):
                        json_obj = iterate_image(tokenizers,text_encoders,vae,folder_path,file,caption_exts=caption_exts,recreate_cache=recreate_cache)
                        datarows.append(json_obj)
                        
        
        # Serializing json
        json_object = json.dumps(datarows, indent=4)
        
        # Writing to metadata.json
        with open(metadata_path, "w", encoding='utf-8') as outfile:
            outfile.write(json_object)
    
    return datarows

def iterate_image(tokenizers,text_encoders,vae,folder_path,file,caption_exts='.txt,.wd14_cap',recreate_cache=False):
    # get filename and ext from file
    filename, _ = os.path.splitext(file)
    image_path = os.path.join(folder_path, file)
    json_obj = {
        'image_path':image_path
    }
    # read caption
    for ext in caption_exts.split(','):
        text_path = os.path.join(folder_path, f'{filename}{ext}')
        # prompt = ''
        if os.path.exists(text_path):
            json_obj["text_path"] = text_path
            # metadata_file.write(read_caption(folder_path,filename,ext))
            json_obj['prompt'] = open(text_path, encoding='utf-8').read()
            # datarows.append(caption)

    json_obj = cache_file(tokenizers,text_encoders,vae,json_obj,recreate=recreate_cache)
    return json_obj

# based on image_path, caption_path, caption create json object
# write tensor related to npz file
def cache_file(tokenizers,text_encoders,vae,json_obj,cache_ext=".nphy",recreate=False):
    image_path = json_obj["image_path"]
    prompt = json_obj["prompt"]
    
    try:
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as e:
        print(f"Failed to open {image_path}: {e}")
        # continue

    ##############################################################################
    # Simple center crop for others
    ##############################################################################
    width, height = image.size
    original_size = (height,width)
    
    rope_img = f"base{BASE_RESOLUTION}"
    buckets = get_buckets()
    resolutions = list(buckets.keys())
    
    # from hunyuan model config {'depth': 40, 'hidden_size': 1408, 'patch_size': 2, 'num_heads': 16, 'mlp_ratio': 4.3637},
    patch_size = 2
    hidden_size = 1408
    num_heads = 16
    # freqs_cis_img = {}
    # for reso in resolutions:
    #     reso_height,reso_width = tuple(reso.split('x'))
    #     reso_height = int(reso_height)
    #     reso_width = int(reso_width)
    #     th, tw = int(reso_height) // 8 // patch_size, int(reso_width) // 8 // patch_size
    #     sub_args = calc_sizes(rope_img, patch_size, th, tw)
    #     freqs_cis_img[str(reso)] = get_2d_rotary_pos_embed(hidden_size // num_heads, *sub_args, use_real=True)
    #     print(f"    Using image RoPE ({rope_img}) ({'real'}): {sub_args} | ({reso}) "
    #            f"{freqs_cis_img[str(reso)][0].shape }")
    # return freqs_cis_img
    
    
    # resolutions = ResolutionGroup(1024,
    #                                 align=16,
    #                                 step=64,
    #                                 target_ratios=["1:1", "3:4", "4:3", "16:9", "9:16"]).data
    # resolutions = [
    #     (1024,1024),
    #     (864,1152),
    #     (1152,864),
    #     (1280,720),
    #     (720,1280),
    # ]
    
    
    horizontal_resolution_set = RESOLUTION_SET
    vertical_resolution_set = [(width,height) for height,width in RESOLUTION_SET]
    resolutions = horizontal_resolution_set + vertical_resolution_set[1:]
    rope_img="base512"
    patch_size=2
    hidden_size=1408
    num_heads=16
    rope_real=True
    freqs_cis_img = {}
    for height,width in resolutions:
        th, tw = height // 8 // patch_size, width // 8 // patch_size
        sub_args = calc_sizes(rope_img, patch_size, th, tw)
        freqs_cis_img[f"{height}x{width}"] = get_2d_rotary_pos_embed(hidden_size // num_heads, *sub_args, use_real=rope_real)
        # print(f"    Using image RoPE ({rope_img}) ({'real' if rope_real else 'complex'}): {sub_args} | ({height}x{width}) "
        #        f"{freqs_cis_img[f"{height}x{width}"][0].shape if rope_real else freqs_cis_img[f"{height}x{width}"].shape}")
    
    image = numpy.array(image)
    # get nearest resolution
    closest_ratio,closest_resolution = get_nearest_resolution(image)
    # we need to expand the closest resolution to target resolution before cropping
    scale_ratio = closest_resolution[0] / closest_resolution[1]
    # image_ratio = width / height
    image_ratio = height / width
    scale_with_height = True
    # referenced kohya ss code
    if image_ratio < scale_ratio: 
        scale_with_height = False
    try:
        image,crop_x,crop_y = simple_center_crop(image,scale_with_height,closest_resolution)
    except Exception as e:
        print(e)
        raise e
    
    # test = Image.fromarray(image)
    # test.show()
    # set meta data
    image_height, image_width, _ = image.shape
    target_size = (image_height,image_width)
    ##############################################################################
    
    # json_obj['bucket'] = f"{image_width}x{image_height}"
    json_obj['bucket'] = f"{image_height}x{image_width}"
    

    file_path,_ = os.path.splitext(image_path)
    npz_path = f'{file_path}{cache_ext}'

    json_obj["npz_path"] = npz_path
    # check if file exists
    if os.path.exists(npz_path):
        # load file via torch
        try:
            if recreate:
                # remove the cache
                os.remove(npz_path)
            else:
                # not need to load embedding. it would load while training
                # embedding = torch.load(npz_path)
                return json_obj
        except Exception as e:
            print(e)
            print(f"{npz_path} is corrupted, regenerating...")
    
    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    image = train_transforms(image)
    
    # create tensor latent
    # pixel_values = image.to(memory_format=torch.contiguous_format).to(vae.device, dtype=vae.dtype).unsqueeze(0)
    pixel_values = []
    pixel_values.append(image)
    pixel_values = torch.stack(pixel_values).to(vae.device)
    del image

    
    with torch.no_grad():
        #contiguous_format = (contiguous memory block), unsqueeze(0) adds bsz 1 dimension, else error: but got weight of shape [128] and input of shape [128, 512, 512]
        latent = vae.encode(pixel_values).latent_dist.sample().squeeze(0)
        # .squeeze(0) #squeeze to remove bsz dimension
        latent = latent * vae.config.scaling_factor
        del pixel_values
        
        cos_cis_img, sin_cis_img = freqs_cis_img[json_obj['bucket']]
        # image_meta_size = [origin_size + target_size + (crop_y,crop_x)]
        image_meta_size = tuple(original_size) + tuple(target_size) + tuple((crop_y,crop_x))
        kwargs = {
            'image_meta_size': image_meta_size,
            'style':0,
        }
        kwargs = {k: torch.tensor(np.array(v)).clone().detach() for k, v in kwargs.items()}
        
        clip_prompt_embeds, clip_attention_masks, t5_prompt_embeds,t5_attention_masks = compute_text_embeddings(text_encoders,tokenizers,prompt,device=vae.device)
        clip_prompt_embed = clip_prompt_embeds.squeeze(0)
        clip_attention_mask = clip_attention_masks.squeeze(0)
        t5_prompt_embed = t5_prompt_embeds.squeeze(0)
        t5_attention_mask = t5_attention_masks.squeeze(0)
        
        latent_dict = dict(
            latent=latent,
            encoder_hidden_state=clip_prompt_embed,
            text_embedding_mask=clip_attention_mask,
            encoder_hidden_state_t5=t5_prompt_embed,
            text_embedding_mask_t5=t5_attention_mask,
            image_meta_size=kwargs['image_meta_size'],
            style=kwargs['style'],
            cos_cis_img=cos_cis_img,
            sin_cis_img=sin_cis_img,
        )
        
        
        # save latent to cache file
        torch.save(latent_dict, npz_path)
        del latent_dict
    return json_obj


def compute_text_embeddings(text_encoders, tokenizers, prompt, device):
    with torch.no_grad():
        clip_prompt_embeds, clip_attention_masks, t5_prompt_embeds,t5_attention_masks = encode_prompt(text_encoders, tokenizers, prompt, device=device)
        clip_prompt_embeds = clip_prompt_embeds.to(device)
        clip_attention_masks = clip_attention_masks.to(device)
        t5_prompt_embeds = t5_prompt_embeds.to(device)
        t5_attention_masks = t5_attention_masks.to(device)
    return clip_prompt_embeds, clip_attention_masks, t5_prompt_embeds,t5_attention_masks


def get_empty_embedding(cache_path="cache/empty_embedding_hy.nphy"):
    if os.path.exists(cache_path):
        return torch.load(cache_path)
def create_empty_embedding(tokenizers,text_encoders,cache_path="cache/empty_embedding_hy.nphy",recreate=False):
    if recreate:
        os.remove(cache_path)

    if os.path.exists(cache_path):
        return torch.load(cache_path)

    freqs_cis_img = {}
    th, tw = 1024 // 8 // 2, 1024 // 8 // 2
    rope_img = "base512"
    reso = "1024x1024"
    sub_args = calc_sizes(rope_img, 2, th, tw)
    freqs_cis_img[str(reso)] = get_2d_rotary_pos_embed(1408 // 16, *sub_args, use_real=True)
    # print(f"    Using image RoPE ({rope_img}) ({'real'}): {sub_args} | ({reso}) "
    #         f"{freqs_cis_img[str(reso)][0].shape }")
    
    image_meta_size = (1024,1024) + (1024,1024) + (0,0)
    kwargs = {
        'image_meta_size': image_meta_size,
        'style':0
    }
    kwargs = {k: torch.tensor(np.array(v)).clone().detach() for k, v in kwargs.items()}
    cos_cis_img, sin_cis_img = freqs_cis_img[reso]
    clip_prompt_embeds, clip_attention_masks, t5_prompt_embeds,t5_attention_masks = compute_text_embeddings(text_encoders,tokenizers,"","cuda")
    clip_prompt_embeds = clip_prompt_embeds.squeeze(0)
    clip_attention_masks = clip_attention_masks.squeeze(0)
    t5_prompt_embeds = t5_prompt_embeds.squeeze(0)
    t5_attention_masks = t5_attention_masks.squeeze(0)
    latent = dict(
        encoder_hidden_state=clip_prompt_embeds,
        text_embedding_mask=clip_attention_masks,
        encoder_hidden_state_t5=t5_prompt_embeds,
        text_embedding_mask_t5=t5_attention_masks,
        image_meta_size=kwargs['image_meta_size'],
        style=kwargs['style'],
        cos_cis_img=cos_cis_img,
        sin_cis_img=sin_cis_img,
    )
    # save latent to cache file
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
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    cpu_offload=False,
):
    
    def fill_t5_token_mask(fill_tensor, fill_number, setting_length):
        fill_length = setting_length - fill_tensor.shape[1]
        if fill_length > 0:
            fill_tensor = torch.cat((fill_tensor, fill_number * torch.ones(1, fill_length)), dim=1)
        return fill_tensor

    # cpu offload the t5
    if cpu_offload:
        device = "cpu"
    text_encoder.to(device)
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_ctx_len_t5 = 256
    text_inputs = tokenizer(
        prompt,
        max_length=text_ctx_len_t5,
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids=fill_t5_token_mask(text_inputs["input_ids"], fill_number=1, setting_length=text_ctx_len_t5).long()
    attention_mask=fill_t5_token_mask(text_inputs["attention_mask"], fill_number=0, setting_length=text_ctx_len_t5).bool()
    
    text_input_ids = text_input_ids.to(device).squeeze(1)
    attention_mask = attention_mask.to(device).squeeze(1)
    with torch.no_grad():
        # text_input_ids = text_inputs.input_ids
        output_t5 = text_encoder(
            text_input_ids,
            attention_mask=attention_mask if T5_ENCODER['attention_mask'] else None,
            output_hidden_states=True
            )
        encoder_hidden_states_t5 = output_t5['hidden_states'][T5_ENCODER['layer_index']].detach()

        # dtype = text_encoder.dtype
        prompt_embeds = encoder_hidden_states_t5.to(device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, attention_mask


def encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    text_embedding = text_inputs.input_ids
    attention_mask = text_inputs.attention_mask

    prompt_embeds = text_encoder(
        text_embedding.to(device),
        attention_mask=attention_mask.to(device),
    )[0]
    
    # pooled_prompt_embeds = prompt_embeds[0]
    # prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, attention_mask


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_text_encoder = text_encoders[0]
    clip_tokenizer = tokenizers[0]

    clip_prompt_embeds, clip_attention_masks = encode_prompt_with_clip(
        text_encoder=clip_text_encoder,
        tokenizer=clip_tokenizer,
        prompt=prompt,
        device=device if device is not None else clip_text_encoder.device,
        num_images_per_prompt=num_images_per_prompt,
    )

    t5_text_encoder = text_encoders[1]
    t5_tokenizer = tokenizers[1]
    t5_prompt_embeds,t5_attention_masks = encode_prompt_with_t5(
        t5_text_encoder,
        t5_tokenizer,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    return clip_prompt_embeds, clip_attention_masks, t5_prompt_embeds,t5_attention_masks


def simple_center_crop(image,scale_with_height,closest_resolution):
    height, width, _ = image.shape
    # width, height = image.size
    
    # print("ori size:",width,height)
    if scale_with_height: 
        # up_scale = height / closest_resolution[1]
        up_scale = width / closest_resolution[1]
    else:
        # up_scale = width / closest_resolution[0]
        up_scale = height / closest_resolution[0]

    expanded_closest_size = (int(closest_resolution[0] * up_scale + 0.5), int(closest_resolution[1] * up_scale + 0.5))
    
    # diff_x = abs(expanded_closest_size[0] - width)
    # diff_y = abs(expanded_closest_size[1] - height)
    diff_x = abs(expanded_closest_size[1] - width)
    diff_y = abs(expanded_closest_size[0] - height)
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

    height, width, _ = cropped_image.shape  
    # due to the resize parameter accept new widht, new height
    # need to re-asign the closest_resolution[1] which width first
    return resize(cropped_image,(closest_resolution[1],closest_resolution[0])),crop_x,crop_y


def resize(img,resolution):
    # return cv2.resize(img,resolution,interpolation=cv2.INTER_AREA)
    return cv2.resize(img,resolution)


#################################################################################
#                   Rotary Positional Embedding Functions                       #
#################################################################################
# https://github.com/facebookresearch/llama/blob/main/llama/model.py#L443

def get_2d_rotary_pos_embed(embed_dim, start, *args, use_real=True):
    """
    This is a 2d version of precompute_freqs_cis, which is a RoPE for image tokens with 2d structure.

    Parameters
    ----------
    embed_dim: int
        embedding dimension size
    start: int or tuple of int
        If len(args) == 0, start is num; If len(args) == 1, start is start, args[0] is stop, step is 1;
        If len(args) == 2, start is start, args[0] is stop, args[1] is num.
    use_real: bool
        If True, return real part and imaginary part separately. Otherwise, return complex numbers.

    Returns
    -------
    pos_embed: torch.Tensor
        [HW, D/2]
    """
    grid = get_meshgrid(start, *args)   # [2, H, w]
    grid = grid.reshape([2, 1, *grid.shape[1:]])   # Returns a sampling matrix with the same resolution as the target resolution
    pos_embed = get_2d_rotary_pos_embed_from_grid(embed_dim, grid, use_real=use_real)
    return pos_embed

def get_meshgrid(start, *args):
    if len(args) == 0:
        # start is grid_size
        num = _to_tuple(start)
        start = (0, 0)
        stop = num
    elif len(args) == 1:
        # start is start, args[0] is stop, step is 1
        start = _to_tuple(start)
        stop = _to_tuple(args[0])
        num = (stop[0] - start[0], stop[1] - start[1])
    elif len(args) == 2:
        # start is start, args[0] is stop, args[1] is num
        start = _to_tuple(start)
        stop = _to_tuple(args[0])
        num = _to_tuple(args[1])
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    grid_h = np.linspace(start[0], stop[0], num[0], endpoint=False, dtype=np.float32)
    grid_w = np.linspace(start[1], stop[1], num[1], endpoint=False, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)   # [2, W, H]
    return grid


def get_2d_rotary_pos_embed_from_grid(embed_dim, grid, use_real=False):
    assert embed_dim % 4 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_rotary_pos_embed(embed_dim // 2, grid[0].reshape(-1), use_real=use_real)  # (H*W, D/4)
    emb_w = get_1d_rotary_pos_embed(embed_dim // 2, grid[1].reshape(-1), use_real=use_real)  # (H*W, D/4)

    if use_real:
        cos = torch.cat([emb_h[0], emb_w[0]], dim=1)    # (H*W, D/2)
        sin = torch.cat([emb_h[1], emb_w[1]], dim=1)    # (H*W, D/2)
        return cos, sin
    else:
        emb = torch.cat([emb_h, emb_w], dim=1)    # (H*W, D/2)
        return emb


def get_1d_rotary_pos_embed(dim: int, pos: Union[np.ndarray, int], theta: float = 10000.0, use_real=False):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        pos (np.ndarray, int): Position indices for the frequency tensor. [S] or scalar
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (bool, optional): If True, return real part and imaginary part separately.
                                   Otherwise, return complex numbers.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials. [S, D/2]

    """
    if isinstance(pos, int):
        pos = np.arange(pos)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # [D/2]
    t = torch.from_numpy(pos).to(freqs.device)  # type: ignore  # [S]
    freqs = torch.outer(t, freqs).float()  # type: ignore   # [S, D/2]
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1)  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1)  # [S, D]
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis



def calc_sizes(rope_img, patch_size, th, tw):
    if rope_img == 'extend':
        # Expansion mode
        sub_args = [(th, tw)]
    elif rope_img.startswith('base'):
        # Based on the specified dimensions, other dimensions are obtained through interpolation.
        base_size = int(rope_img[4:]) // 8 // patch_size
        start, stop = get_fill_resize_and_crop((th, tw), base_size)
        sub_args = [start, stop, (th, tw)]
    else:
        raise ValueError(f"Unknown rope_img: {rope_img}")
    return sub_args

def _to_tuple(x):
    if isinstance(x, int):
        return x, x
    else:
        return x
def get_fill_resize_and_crop(src, tgt):
    th, tw = _to_tuple(tgt)
    h, w = _to_tuple(src)

    tr = th / tw        # base resolution
    r = h / w           # target resolution

    # resize
    if r > tr:
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))    # resize the target resolution down based on the base resolution

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


if __name__ == "__main__":
    image = Image.open("F:/ImageSet/handpick_high_quality/animal/blue-jay-8075346.jpg")
    
    # set meta data
    width, height = image.size
    
    
    open_cv_image = numpy.array(image)
    # # Convert RGB to BGR
    image = open_cv_image[:, :, ::-1].copy()
    
    # get nearest resolution
    closest_ratio,closest_resolution = get_nearest_resolution(image)
    # print('init closest_resolution',closest_resolution)

    # we need to expand the closest resolution to target resolution before cropping
    scale_ratio = closest_resolution[0] / closest_resolution[1]
    image_ratio = width / height

    # scale_with_height = True
    scale_with_height = False
    # referenced kohya ss code
    if image_ratio < scale_ratio: 
        scale_with_height = True
    try:
        image,_,_ = simple_center_crop(image,scale_with_height,closest_resolution)
        # save_webp(simple_crop_image,filename,'simple',os.path.join(output_dir,"simple"))
    except Exception as e:
        print(e)
        raise e
    # set meta data
    image_height, image_width, _ = image.shape