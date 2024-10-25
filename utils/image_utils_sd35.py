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
    get_md5_by_path
)
import glob
from utils.dist_utils import flush
import numpy as np
import pandas as pd


# BASE_RESOLUTION = 1024

# RESOLUTION_SET = [
#     (1024, 1024),
#     (1152, 896),
#     (1216, 832),
#     (1344, 768),
#     (1536, 640),
# ]

RESOLUTION_CONFIG = {
    512: [
        # extra resolution for testing
        # (1536, 1536),
        # (672, 672),
        (512, 512),
        (768, 512), # 1.5
        (576, 448), # 1.5
        (608, 416), # 1.5
    ],
    1024: [
        # extra resolution for testing
        # (1536, 1536),
        # (1344, 1344),
        (1024, 1024),
        (1344, 1024),
        (1152, 896), # 1.2857
        (1216, 832), # 1.46
        (1344, 768), # 1.75
        (1536, 640), # 2.4
    ],
    2048: [
        (2048, 2048),
        (2304, 1792), # 1.2857
        (2432, 1664), # 1.46
        (2688, 1536), # 1.75
        (3072, 1280), # 2.4
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
    height, width, _ = image.shape
    # if height==width and (width <= 1344 and height <= 1344):
    #     closest_pixel = closest_mod_64(width)
    #     return 1, (closest_pixel,closest_pixel)
    
    resolution_set = RESOLUTION_CONFIG[resolution]
    
    # get ratio
    image_ratio = width / height

    horizontal_resolution_set = resolution_set
    horizontal_ratio = [round(width/height, 2) for width,height in resolution_set]

    vertical_resolution_set = [(height,width) for width,height in resolution_set]
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
            # skip leftovers
            elif batch:  #else store leftovers for the next epoch
                # print("leftovers",batch)
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
        embedding = get_empty_embedding()
        self.empty_prompt_embed = embedding['prompt_embed']  # Tuple of (empty_prompt_embed, empty_pooled_prompt_embed)
        self.empty_pooled_prompt_embed = embedding['pooled_prompt_embed']

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
        cached_npz = torch.load(metadata['npz_path'])
        cached_latent = torch.load(metadata['latent_path'])
        latent = cached_latent['latent']
        prompt_embed = cached_npz['prompt_embed']
        pooled_prompt_embed = cached_npz['pooled_prompt_embed']
        # time_id = cached_npz['time_id']

        # conditional_dropout
        if random.random() < self.conditional_dropout_percent:
            prompt_embed = self.empty_prompt_embed
            pooled_prompt_embed = self.empty_pooled_prompt_embed

        return {
            "latent": latent,
            "prompt_embed": prompt_embed,
            "pooled_prompt_embed": pooled_prompt_embed,
            # "time_id": time_id,
        }



##input: datarows -> output: metadata
#looks like leftover code from leftover_idx, check, then delete
class CachedPairsDataset(Dataset):
    def __init__(self, datarows,conditional_dropout_percent=0.1): 
        self.datarows = datarows
        self.leftover_indices = []  #initialize an empty list to store indices of leftover items
        #for conditional_dropout
        self.conditional_dropout_percent = conditional_dropout_percent
        embedding = get_empty_embedding()
        self.empty_prompt_embed = embedding['prompt_embed']  # Tuple of (empty_prompt_embed, empty_pooled_prompt_embed)
        self.empty_pooled_prompt_embed = embedding['pooled_prompt_embed']

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
        pos_npz = torch.load(metadata['pos_npz_path'])
        pos_latent_dict = torch.load(metadata['pos_latent_path'])
        
        pos_prompt_embed = pos_npz['prompt_embed']
        pos_pooled_prompt_embed = pos_npz['pooled_prompt_embed']
        pos_latent = pos_latent_dict['latent']
        # pos_time_id = pos_latent_dict['time_id']
        
        # conditional_dropout
        if random.random() < self.conditional_dropout_percent:
            pos_prompt_embed = self.empty_prompt_embed
            pos_pooled_prompt_embed = self.empty_pooled_prompt_embed

        
        neg_npz = torch.load(metadata['neg_npz_path'])
        neg_latent_dict = torch.load(metadata['neg_latent_path'])
        
        neg_prompt_embed = neg_npz['prompt_embed']
        neg_pooled_prompt_embed = neg_npz['pooled_prompt_embed']
        neg_latent = neg_latent_dict['latent']
        # neg_time_id = neg_latent_dict['time_id']
        
        # conditional_dropout
        if random.random() < self.conditional_dropout_percent:
            neg_prompt_embed = self.empty_prompt_embed
            neg_pooled_prompt_embed = self.empty_pooled_prompt_embed
        
        main_npz = torch.load(metadata['main_npz_path'])
        main_prompt_embed = main_npz['prompt_embed']
        main_pooled_prompt_embed = main_npz['pooled_prompt_embed']
        
        # uncondition_npz = torch.load(metadata['uncondition_npz_path'])
        # uncondition_prompt_embed = uncondition_npz['prompt_embed']
        # uncondition_pooled_prompt_embed = uncondition_npz['pooled_prompt_embed']

        return {
            # positive latent
            "pos_latent": pos_latent,
            "pos_prompt_embed": pos_prompt_embed,
            "pos_pooled_prompt_embed": pos_pooled_prompt_embed,
            # "pos_time_id":pos_time_id,
            
            # negative latent
            "neg_latent": neg_latent,
            "neg_prompt_embed": neg_prompt_embed,
            "neg_pooled_prompt_embed": neg_pooled_prompt_embed,
            # "neg_time_id":neg_time_id,
            
            # uncondition embedding
            "main_prompt_embed": main_prompt_embed,
            "main_pooled_prompt_embed": main_pooled_prompt_embed,
        }

# main idea is store all tensor related in .npz file
# other information stored in .json
@torch.no_grad()
def create_metadata_cache(tokenizers,text_encoders,vae,image_files,recreate_cache=False, metadata_path="metadata_sd35.json", resolution_config="1024"):
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
            resolutions=resolutions,recreate_cache=recreate_cache)
        
        embedding_objects.append(json_obj)
    
    # move glm to cpu to reduce vram memory
    text_encoders[0].to("cpu")
    del text_encoders
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
def create_embedding(tokenizers,text_encoders,folder_path,file,cache_ext=".npsd35",resolutions=None,recreate_cache=False):
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
    json_obj["npz_path"] = npz_path
    
    if not recreate_cache and os.path.exists(npz_path):
        if 'npz_path_md5' not in json_obj:
            json_obj["npz_path_md5"] = get_md5_by_path(npz_path)
        return json_obj
    
    prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(text_encoders,tokenizers,content,device=text_encoders[0].device)
    prompt_embed = prompt_embeds.squeeze(0)
    pooled_prompt_embed = pooled_prompt_embeds.squeeze(0)
    
    try:
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is not None:
            # Convert to RGB format (assuming the original image is in BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            print(f"Failed to open {image_path}.")
    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")

    height, width, _ = image.shape
    original_size = (height, width)
    crops_coords_top_left = (0,0)
    # time_id = torch.tensor(list(original_size + crops_coords_top_left + original_size)).to(text_encoders[0].device, dtype=text_encoders[0].dtype)
    npz_dict = {
        "prompt_embed": prompt_embed.cpu(), 
        "pooled_prompt_embed": pooled_prompt_embed.cpu(),
        # "time_id": time_id.cpu()
    }
    
    # save latent to cache file
    torch.save(npz_dict, npz_path)
    return json_obj

# based on image_path, caption_path, caption create json object
# write tensor related to npz file
@torch.no_grad()
def cache_file(vae,json_obj,resolution=1024,cache_ext=".npsd35",latent_ext=".npsd35latent",recreate_cache=False):
    npz_path = json_obj["npz_path"]
    
    
    latent_cache_path = npz_path.replace(cache_ext,latent_ext)
    if resolution > 1024:
        latent_cache_path = npz_path.replace(cache_ext,f"_{resolution}{latent_ext}")
    json_obj["latent_path"] = latent_cache_path
    
    
    npz_dict = {}
    if os.path.exists(npz_path):
        try:
            npz_dict = torch.load(npz_path)
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
    original_size = (height, width)
    
    # get nearest resolution
    closest_ratio,closest_resolution = get_nearest_resolution(image,resolution=resolution)
    # we need to expand the closest resolution to target resolution before cropping
    scale_ratio = closest_resolution[0] / closest_resolution[1]
    image_ratio = width / height
    scale_with_height = True
    crops_coords_top_left = (0,0)
    # referenced kohya ss code
    if image_ratio < scale_ratio: 
        scale_with_height = False
    try:
        # image = simple_center_crop(image,scale_with_height,closest_resolution)
        image,crop_x,crop_y = simple_center_crop(image,scale_with_height,closest_resolution)
        crops_coords_top_left = (crop_y,crop_x)
        # save_webp(simple_crop_image,filename,'simple',os.path.join(output_dir,"simple"))
    except Exception as e:
        print(e)
        raise e
    # test = Image.fromarray(image)
    # test.show()
    # set meta data
    image_height, image_width, _ = image.shape
    target_size = (image_height,image_width)
    ##############################################################################
    
    json_obj['bucket'] = f"{image_width}x{image_height}"
    
    # time_id = torch.tensor(list(original_size + crops_coords_top_left + target_size)).to(vae.device, dtype=vae.dtype)

    # skip if already cached
    if os.path.exists(latent_cache_path) and not recreate_cache:
        if 'latent_path_md5' not in json_obj:
            json_obj['latent_path_md5'] = get_md5_by_path(latent_cache_path)
            json_obj['npz_path_md5'] = get_md5_by_path(npz_path)
        return json_obj
    
    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
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
        latent = latent * vae.config.scaling_factor
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

def compute_text_embeddings(text_encoders, tokenizers, prompt, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, device=device)
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def get_empty_embedding(cache_path="cache/empty_embedding.npsd35"):
    if os.path.exists(cache_path):
        return torch.load(cache_path)
def create_empty_embedding(tokenizers,text_encoders,cache_path="cache/empty_embedding.npsd35",recreate=False, resolution=1024):
    if recreate:
        os.remove(cache_path)

    if os.path.exists(cache_path):
        return torch.load(cache_path)

    prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders,tokenizers,"")
    prompt_embeds = prompt_embeds.squeeze(0)
    pooled_prompt_embeds = pooled_prompt_embeds.squeeze(0)
    # time_id = torch.tensor([
    #     # original size
    #     resolution,resolution,
    #     0,0,
    #     # target size
    #     resolution,resolution
    # ])
    latent = {
        "prompt_embed": prompt_embeds.cpu(), 
        "pooled_prompt_embed": pooled_prompt_embeds.cpu(),
        # "time_id":time_id.cpu()
    }
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

    return prompt_embeds


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

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds

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
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
        prompt_embeds, pooled_prompt_embeds = encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        256,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds
    
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


def crop_image(image,resolution=1024):
    height, width, _ = image.shape
    # get nearest resolution
    closest_ratio,closest_resolution = get_nearest_resolution(image,resolution=resolution)
    # we need to expand the closest resolution to target resolution before cropping
    scale_ratio = closest_resolution[0] / closest_resolution[1]
    image_ratio = width / height
    scale_with_height = True
    # referenced kohya ss code
    if image_ratio < scale_ratio: 
        scale_with_height = False
    try:
        image,crop_x,crop_y = simple_center_crop(image,scale_with_height,closest_resolution)
    except Exception as e:
        print(e)
        raise e
    
    return image,crop_x,crop_y

def resize(img,resolution):
    # return cv2.resize(img,resolution,interpolation=cv2.INTER_AREA)
    return cv2.resize(img,resolution, interpolation=cv2.INTER_CUBIC)

    