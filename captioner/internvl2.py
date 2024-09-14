"""
This is a demo for using CogAgent and CogVLM in CLI
Make sure you have installed vicuna-7b-v1.5 tokenizer model (https://huggingface.co/lmsys/vicuna-7b-v1.5), full checkpoint of vicuna-7b-v1.5 LLM is not required.
In this demo, We us chat template, you can use others to replace such as 'vqa'.
Strongly suggest to use GPU with bfloat16 support, otherwise, it will be slow.
Mention that only one picture can be processed at one conversation, which means you can not replace or insert another picture during the conversation.
"""

import argparse
import torch

from PIL import Image
from transformers import AutoTokenizer, AutoModel
import os
import time
from ModelWrapper import ModelWrapper
from utils import flush,get_device
import glob
from tqdm import tqdm
import cv2
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import shutil

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

class InternVL2ModelWrapper(ModelWrapper):

    def __init__(self,device=None,dtype=None,tokenizer_repo_id="lmsys/vicuna-7b-v1.5"):
        super().__init__()
        self.device = get_device(device)
        self.model_repo_id = 'OpenGVLab/InternVL2-2B'
        self.tokenizer_repo_id = tokenizer_repo_id
        if dtype == None:
            self.dtype = torch.bfloat16
        else:
            self.dtype = dtype
        # self.prompt = f'Describe the image precisely, detailing every element, interaction and background. Include composition, angle and perspective. Use only facts and concise language; avoid interpretations or speculation:'
        self.prompt = '<image>\nPlease describe the image shortly.'
        self.starts_with = f'The image showcases '
        self.model = AutoModel.from_pretrained(
                self.model_repo_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_repo_id, trust_remote_code=True, use_fast=False)

        
        # self.generation_config = dict(max_new_tokens=1024, do_sample=True)
        
    def execute(self, image=None,prompt=None,starts_with=None, image_path=None):
        model = self.model
        tokenizer = self.tokenizer
        if prompt != None:
            self.prompt = prompt
        if starts_with != None:
            self.starts_with = starts_with
        
        # set the max number of tiles in `max_num`
        pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens=1024, do_sample=True)

        # single-image single-round conversation (单图单轮对话)
        # question = '<image>\nPlease describe the image shortly.'
        response = model.chat(tokenizer, pixel_values, self.prompt, generation_config)
        # print(f'User: {question}\nAssistant: {response}')

        
        del pixel_values
        return response
       
if __name__ == "__main__":
    input_dir = "F:/ImageSet/kolors_pony/training/character"
    model = InternVL2ModelWrapper()
    prefix = "二次元动漫风格, anime artwork, "
    max_attempt_count = 3
    character = "character_name"
    
    files = glob.glob(f"{input_dir}/**", recursive=True)
    image_exts = [".png",".jpg",".jpeg",".webp"]
    image_files = [f for f in files if os.path.splitext(f)[-1].lower() in image_exts]
    
    
    for image_file in tqdm(image_files,position=2):
        text_file = os.path.splitext(image_file)[0] + ".txt"
        attempt_count = 0
        result = model.execute(image_path=image_file)
        if "I'm sorry, but I " in result:
            while "I'm sorry, but I " in result and attempt_count < max_attempt_count:
                result = model.execute(image_path=image_file)
                attempt_count = attempt_count + 1
        
        new_content = f"{prefix}{character}, {result}"
        # new caption
        with open(text_file, "w", encoding="utf-8") as new_f:
            new_f.write(new_content)
            print("save new caption: ", text_file)