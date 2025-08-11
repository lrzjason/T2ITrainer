from torch.utils.data import Dataset, Sampler
import random
import json
import torch
import os
from torchvision import transforms
from PIL import Image, ImageOps
from tqdm import tqdm 
import gc

MAX_LENGTH = 300

BASE_RESOLUTION = 1024

# RESOLUTION_SET = [
#     (1024, 1024),
#     (1152, 896),
#     (1216, 832),
#     (1344, 768),
#     (1536, 640),
# ]

# height / width for pixart
RESOLUTION_SET = [
    (1024, 1024),
    (896, 1152),
    (832, 1216),
    (768, 1344),
    (640, 1536),
]

def get_buckets():
    horizontal_resolution_set = RESOLUTION_SET
    vertical_resolution_set = [(width,height) for height,width in RESOLUTION_SET]
    all_resolution_set = horizontal_resolution_set + vertical_resolution_set[1:]
    buckets = {}
    for resolution in all_resolution_set:
        buckets[f'{resolution[0]}x{resolution[1]}'] = []
    return buckets

# return closest_ratio and width,height closest_resolution
def get_nearest_resolution(image):
    _, height, width = image.shape
    
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
        self.empty_prompt_embed_mask = embedding['prompt_embed_mask']
        self.empty_data_info = embedding['data_info']

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
        prompt_embed = cached_latent['prompt_embed']
        prompt_embed_mask = cached_latent['prompt_embed_mask']
        data_info = cached_latent['data_info']
        # meta_index = actual_index

        #conditional_dropout
        if random.random() < self.conditional_dropout_percent:
            prompt_embed = self.empty_prompt_embed
            prompt_embed_mask = self.empty_prompt_embed_mask
            data_info = self.empty_data_info

        return {
            "latent": latent,
            "prompt_embed": prompt_embed,
            "prompt_embed_mask": prompt_embed_mask,
            "data_info": data_info
            # ,
            # "meta_index": meta_index
        }
    
# main idea is store all tensor related in .npz file
# other information stored in .json
def create_metadata_cache(tokenizer,text_encoder,vae,input_dir,caption_exts='.txt,.wd14_cap',recreate=False,recreate_cache=False):
    create_empty_embedding(tokenizer,text_encoder)
    supported_image_types = ['.jpg','.png','.webp']
    metadata_path = os.path.join(input_dir, 'metadata.json')
    if recreate:
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
                            json_obj = iterate_image(tokenizer,text_encoder,vae,folder_path,file,caption_exts=caption_exts,recreate_cache=recreate_cache)
                            datarows.append(json_obj)
            # handle single files
            else:
                folder_path = input_dir
                file = item
                for image_type in supported_image_types:
                    if file.endswith(image_type):
                        json_obj = iterate_image(tokenizer,text_encoder,vae,folder_path,file,caption_exts=caption_exts,recreate_cache=recreate_cache)
                        datarows.append(json_obj)
        
        # Serializing json
        json_object = json.dumps(datarows, indent=4)
        
        # Writing to metadata.json
        with open(metadata_path, "w", encoding='utf-8') as outfile:
            outfile.write(json_object)
    
    return datarows

def iterate_image(tokenizer,text_encoder,vae,folder_path,file,caption_exts='.txt,.wd14_cap',recreate_cache=False):
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

    json_obj = cache_file(tokenizer,text_encoder,vae,json_obj,recreate=recreate_cache)
    return json_obj

# based on image_path, caption_path, caption create json object
# write tensor related to npz file
def cache_file(tokenizer,text_encoder,vae,json_obj,cache_ext=".npz",recreate=False):

    image_path = json_obj["image_path"]
    prompt = json_obj["prompt"]
    
    try:
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as e:
        print(f"Failed to open {image_path}: {e}")
        # continue

    # set meta data
    image_width, image_height = image.size
    json_obj['bucket'] = f"{image_width}x{image_height}"
    

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
    
    # all images are preprocessed to target size, so it doesn't have crop_top_left
    # json_obj["original_size"] = (image_height,image_width)
    # json_obj["crop_top_left"] = (0,0)

    from utils.utils import ToTensorUniversal
    train_transforms = transforms.Compose([ToTensorUniversal(), transforms.Normalize([0.5], [0.5])])
    image = train_transforms(image)
    
    # create tensor latent
    pixel_values = image.to(memory_format=torch.contiguous_format).to(vae.device, dtype=vae.dtype).unsqueeze(0)
    closest_ratio,closest_resolution = get_nearest_resolution(image)
    del image

    
    with torch.no_grad():
        #contiguous_format = (contiguous memory block), unsqueeze(0) adds bsz 1 dimension, else error: but got weight of shape [128] and input of shape [128, 512, 512]
        latent = vae.encode(pixel_values).latent_dist.sample().squeeze() #squeeze to remove bsz dimension
        del pixel_values
        latent = latent * vae.config.scaling_factor
        # print(latent.shape) torch.Size([4, 144, 112])

    prompt_tokens = tokenizer(
        prompt, max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt",
        padding_side="right"
    ).to(text_encoder.device)
    prompt_embed = text_encoder(prompt_tokens.input_ids, attention_mask=prompt_tokens.attention_mask)[0]

    data_info = {}
    data_info['img_hw'] = torch.tensor([image_height, image_width], dtype=torch.float32)
    data_info['aspect_ratio'] = closest_ratio
    latent_dict = {
        "latent": latent.cpu(),
        "prompt_embed": prompt_embed.cpu(), 
        "prompt_embed_mask": prompt_tokens.attention_mask.cpu(),
        "data_info": data_info
        # "img_hw": torch.tensor([image_height, image_width], dtype=torch.float32),
        # "aspect_ratio":closest_ratio
    }

    
    # save latent to cache file
    torch.save(latent_dict, npz_path)
    del prompt_tokens,prompt_embed
    gc.collect()
    torch.cuda.empty_cache()

    return json_obj

def get_empty_embedding(cache_path="cache/pixart_empty_embedding.npz"):
    if os.path.exists(cache_path):
        return torch.load(cache_path)
def create_empty_embedding(tokenizer,text_encoder,cache_path="cache/pixart_empty_embedding.npz",recreate=False):
    # data_info = {}
    # data_info['img_hw'] = torch.tensor([BASE_RESOLUTION, BASE_RESOLUTION], dtype=torch.float32)
    # data_info['aspect_ratio'] = 1.0
    if recreate:
        os.remove(cache_path)

    if os.path.exists(cache_path):
        return torch.load(cache_path)

    null_tokens = tokenizer(
        "", max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt",
        padding_side="right"
    ).to(text_encoder.device)
    null_token_emb = text_encoder(null_tokens.input_ids, attention_mask=null_tokens.attention_mask)[0]
    null_emb_dict = {
        'prompt_embed': null_token_emb, 
        'prompt_embed_mask': null_tokens.attention_mask,
        'data_info':{
            'img_hw': torch.tensor([BASE_RESOLUTION, BASE_RESOLUTION], dtype=torch.float32),
            'aspect_ratio':1.0
        }
    }
    # save latent to cache file
    torch.save(null_emb_dict, cache_path)

    return null_emb_dict