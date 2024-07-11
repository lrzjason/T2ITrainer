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

BASE_RESOLUTION = 1024

RESOLUTION_SET = [
    (1024, 1024),
    (1152, 896),
    (1216, 832),
    (1344, 768),
    (1536, 640),
]

def get_buckets():
    horizontal_resolution_set = RESOLUTION_SET
    vertical_resolution_set = [(height,width) for width,height in RESOLUTION_SET]
    all_resolution_set = horizontal_resolution_set + vertical_resolution_set[1:]
    buckets = {}
    for resolution in all_resolution_set:
        buckets[f'{resolution[0]}x{resolution[1]}'] = []
    return buckets

# return closest_ratio and width,height closest_resolution
def get_nearest_resolution(image):
    height, width, _ = image.shape
    
    # get ratio
    image_ratio = width / height

    horizontal_resolution_set = RESOLUTION_SET
    horizontal_ratio = [round(width/height, 2) for width,height in RESOLUTION_SET]

    vertical_resolution_set = [(height,width) for width,height in RESOLUTION_SET]
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
        cached_latent = torch.load(metadata['npz_path'])
        latent = cached_latent['latent']
        prompt_embed = cached_latent['prompt_embed']
        pooled_prompt_embed = cached_latent['pooled_prompt_embed']
        # time_id = cached_latent['time_id']

        #conditional_dropout
        # if random.random() < self.conditional_dropout_percent:
        #     prompt_embed = self.empty_prompt_embed
        #     pooled_prompt_embed = self.empty_pooled_prompt_embed

        return {
            "latent": latent,
            "prompt_embed": prompt_embed,
            "pooled_prompt_embed": pooled_prompt_embed,
            # "time_id": time_id,
        }
        # return {
        #     "npz_path":metadata['npz_path']
        # }
    
# main idea is store all tensor related in .npz file
# other information stored in .json
def create_metadata_cache(tokenizers,text_encoders,vae,input_dir,caption_exts='.txt,.wd14_cap',recreate=False,recreate_cache=False,  metadata_name="metadata_sd3.json"):
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
def cache_file(tokenizers,text_encoders,vae,json_obj,cache_ext=".npsd3",recreate=False):

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
    image = numpy.array(image)
    # get nearest resolution
    closest_ratio,closest_resolution = get_nearest_resolution(image)
    # we need to expand the closest resolution to target resolution before cropping
    scale_ratio = closest_resolution[0] / closest_resolution[1]
    image_ratio = width / height
    scale_with_height = True
    # referenced kohya ss code
    if image_ratio < scale_ratio: 
        scale_with_height = False
    try:
        image = simple_center_crop(image,scale_with_height,closest_resolution)
        # save_webp(simple_crop_image,filename,'simple',os.path.join(output_dir,"simple"))
    except Exception as e:
        print(e)
        raise e
    # test = Image.fromarray(image)
    # test.show()
    # set meta data
    image_height, image_width, _ = image.shape
    ##############################################################################
    
    
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
        # print(latent.shape) torch.Size([4, 144, 112])

    prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(text_encoders,tokenizers,prompt,device=vae.device)
    prompt_embed = prompt_embeds.squeeze(0)
    pooled_prompt_embed = pooled_prompt_embeds.squeeze(0)
    
    latent_dict = {
        "latent": latent.cpu(),
        "prompt_embed": prompt_embed.cpu(), 
        "pooled_prompt_embed": pooled_prompt_embed.cpu(),
        # "time_id": time_id.cpu()
    }

    
    # save latent to cache file
    torch.save(latent_dict, npz_path)
    del latent_dict
    return json_obj


def compute_text_embeddings(text_encoders, tokenizers, prompt, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, device=device)
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def get_empty_embedding(cache_path="cache/empty_embedding_sd3.npsd3"):
    if os.path.exists(cache_path):
        return torch.load(cache_path)
def create_empty_embedding(tokenizers,text_encoders,cache_path="cache/empty_embedding_sd3.npsd3",recreate=False):
    if recreate:
        os.remove(cache_path)

    if os.path.exists(cache_path):
        return torch.load(cache_path)

    prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders,tokenizers,"")
    prompt_embeds = prompt_embeds.squeeze(0)
    pooled_prompt_embeds = pooled_prompt_embeds.squeeze(0)
    latent = {
        "prompt_embed": prompt_embeds.cpu(), 
        "pooled_prompt_embed": pooled_prompt_embeds.cpu()
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
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    cpu_offload=False,
):
    # cpu offload the t5
    if cpu_offload:
        device = "cpu"
    text_encoder.to(device)
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds.to("cuda")


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

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


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

    print(f"ori ratio:{width/height}")
    height, width, _ = cropped_image.shape  
    print(f"cropped ratio:{width/height}")
    print(f"closest ratio:{closest_resolution[0]/closest_resolution[1]}")
    # resize image to target resolution
    # return cv2.resize(cropped_image, closest_resolution)
    return resize(cropped_image,closest_resolution)


def resize(img,resolution):
    # return cv2.resize(img,resolution,interpolation=cv2.INTER_AREA)
    return cv2.resize(img,resolution)

if __name__ == "__main__":
    image = Image.open("F:/ImageSet/handpick_high_quality/animal/blue-jay-8075346.jpg")
    
    # set meta data
    width, height = image.size
    
    
    open_cv_image = numpy.array(image)
    # # Convert RGB to BGR
    # image = open_cv_image[:, :, ::-1].copy()
    
    # get nearest resolution
    closest_ratio,closest_resolution = get_nearest_resolution(image)
    # print('init closest_resolution',closest_resolution)

    # we need to expand the closest resolution to target resolution before cropping
    scale_ratio = closest_resolution[0] / closest_resolution[1]
    image_ratio = width / height

    scale_with_height = True
    # referenced kohya ss code
    if image_ratio < scale_ratio: 
        scale_with_height = False
    try:
        image = simple_center_crop(image,scale_with_height,closest_resolution)
        # save_webp(simple_crop_image,filename,'simple',os.path.join(output_dir,"simple"))
    except Exception as e:
        print(e)
        raise e
    # set meta data
    image_height, image_width, _ = image.shape