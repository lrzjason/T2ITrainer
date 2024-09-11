import os
import csv
import torch
import numpy as np
import pandas as pd
import onnxruntime
from PIL import Image
import cv2
from pathlib import Path
from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import re
import gc

from ModelWrapper import ModelWrapper
from utils import flush
import random
from PIL import ImageOps
import PIL

import glob
def download_model_files(model_repo_id):
    # Define local paths to save the files
    local_model_path = hf_hub_download(repo_id=model_repo_id, filename='model.onnx')
    local_tags_path = hf_hub_download(repo_id=model_repo_id, filename='selected_tags.csv')

    return local_model_path, local_tags_path

def clean_text(text):
    return ''.join([char if ord(char) < 128 else '' for char in text])

def handle_character_name(text):
    # clear_text = remove_tag_prefix(text)
    text = text.replace("\\","").replace("(","_").replace(")","").replace(" ","_").replace(",","_").replace(":","_")
    text = text.replace("__","_")
    return clean_text(text)
# based on file size, return lossless, quality
def get_webp_params(filesize_mb):
    if filesize_mb <= 2:
        return (False, 100)
    if filesize_mb <= 4:
        return (False, 90)
    return (False, 80)

# Set the maximum pixels to prevent out of memory error
PIL.Image.MAX_IMAGE_PIXELS = 933120000
def preprocess_image(image):
    image = image.convert('RGBA')
    bg = Image.new('RGBA', image.size, 'WHITE')
    bg.paste(image, mask=image)
    image = bg.convert('RGB')
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # convert to BGR format
    h, w = image.shape[:2]
    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    image = np.pad(image, [(pad_h, pad_h), (pad_w, pad_w), (0, 0)], mode='constant', constant_values=255)
    image = cv2.resize(image, (448, 448), interpolation=cv2.INTER_AREA)
    image = np.expand_dims(image, 0)
    return image.astype(np.float32)


class WD14ModelWrapper(ModelWrapper):
    def __init__(self):
        super().__init__()
        self.model_repo_id = 'SmilingWolf/wd-swinv2-tagger-v3'
        model_path, tags_path = download_model_files(self.model_repo_id)
        self.model_path = model_path
        self.tags_path = tags_path
        self.tag_only = True
        self.character_category = 4
        self.model = onnxruntime.InferenceSession(self.model_path, providers=['CUDAExecutionProvider'])
        self.enable_character_caption = True
        self.characteristic_tags = [
            '_hair',
            '_shirt',
            '_skirt',
            '_sleeve',
            'hair_ornament',
            '_glove',
            '_dress',
            '_hat',
            '_ears',
            '_ribbon',
            'twintails',
            'blue_eyes',
            'red_eyes',
            'pink_eyes',
            'black_eyes',
            'aqua_eyes',
            'grey_eyes',
            'orange_eyes',
            'brown_eyes',
            'green_eyes',
            'purple_eyes',
            'yellow_eyes',
            'white_eyes',
            'blank_eyes',
            'extra_eyes',
            'multicolored_eyes',
            'eyeshadow'
        ]
        
        self.gender_tags = [
            '1boy',
            '2boys',
            '3boys',
            '4boys',
            '5boys',
            '6+boys',
            'multiple_boys',
            
            '1girl',
            '2girls',
            '3girls',
            '4girls',
            '5girls',
            '6+girls',
            'multiple_girls',
            
            '1other',
            '2others',
            '3others',
            '4others',
            '5others',
            '6+others',
            'multiple_others'
        ]
        
        self.skip_non_character = True
    def execute(self,image=None,query=None,filter_tags=['questionable','general','sensitive'], tag_threshold=0.7, character_threshold=0.70):
        model = self.model
        tags_scores = []
        processed_image = preprocess_image(image)
        result = model.run(None, {model.get_inputs()[0].name: processed_image})[0]
        tags = pd.read_csv(self.tags_path)
        tags.reset_index(inplace=True)
        result_df = pd.DataFrame(result[0], columns=['Score'])
        result_with_tags = pd.concat([tags, result_df], axis=1)
        tags_filtered = result_with_tags[['name', 'Score', 'category']]
        tags_filtered = tags_filtered[~tags_filtered['name'].isin(filter_tags)]
        tags_scores.append(tags_filtered.set_index('name'))
        averaged_tags_scores = tags_scores[0].reset_index()

        averaged_tags_scores.columns = ['name', 'Score', 'category']  # rename columns
        averaged_tags_scores = averaged_tags_scores[averaged_tags_scores['Score'] > tag_threshold]
        averaged_tags_scores.sort_values('Score', ascending=False, inplace=True)
        
        rows = []
        character_tags = []
        other_tags = []
        skipped_tags = []
        gender_tags = []
        # find character tags
        for _, row in averaged_tags_scores.iterrows():
            tag = row['name']
            if row['category'] == self.character_category:
                print(f"character: {tag} score: {row['Score']}")
                if row['Score'] > character_threshold:
                    character_tags.append(tag)
            else:
                rows.append(row)
        if self.skip_non_character and len(character_tags) == 0:
            return "",gender_tags,character_tags
        for row in rows:
            tag = row['name']
            if row['category'] != self.character_category:
                # when character tag is found and enable_character_caption and tag in characteristic_tags, skip it for character training
                if len(character_tags) > 0 and self.enable_character_caption:
                    found_characteristic_tag = False
                    for characteristic_tag in self.characteristic_tags:
                        if characteristic_tag in tag:
                            # print("skip characteristic tag:", tag)
                            found_characteristic_tag = True
                            break
                    if found_characteristic_tag:
                        # skip characteristic tags
                        skipped_tags.append(tag)
                        continue
                if tag in self.gender_tags:
                    gender_tags.append(tag)
                else:
                    other_tags.append(tag)
                    
        random.shuffle(other_tags)
        all_tags = gender_tags + character_tags + other_tags
        result = ", ".join(all_tags).replace('_',' ')
        
        del averaged_tags_scores,tags_scores,tags_filtered,result_with_tags,result_df,tags,processed_image
        flush()
        
        return result,gender_tags,character_tags

if __name__ == "__main__":
    input_dir = "F:/ImageSet/input_dir"
    output_dir =  "F:/ImageSet/output_dir"
    os.makedirs(output_dir, exist_ok=True)
    
    gender_list = ["male","female","other"]
    for gender in gender_list:
        gender_subdir_path = os.path.join(output_dir, gender)
        os.makedirs(gender_subdir_path, exist_ok=True)
        
    
    files = glob.glob(f"{input_dir}/**", recursive=True)
    image_exts = [".png",".jpg",".jpeg",".webp"]
    image_files = [f for f in files if os.path.splitext(f)[-1].lower() in image_exts]
    model = WD14ModelWrapper()
    # loop input_dir for each image
    for image_path in tqdm(image_files):
        # image_path = os.path.join(input_dir, image_path)
        filename,ext = os.path.splitext(os.path.basename(image_path))
        print(image_path)
        
        possible_text_files = [
            os.path.join(output_dir, "male", filename + ".txt"),
            os.path.join(output_dir, "female", filename + ".txt"),
            os.path.join(output_dir, "other", filename + ".txt")
        ]
        exist_path = ""
        for text_file in possible_text_files:
            if os.path.exists(text_file): 
                exist_path = text_file
                break
        if exist_path != "":
            print(f"{exist_path} exists. Skipped")
            continue
        
        image = Image.open(image_path).convert('RGB')
        result,gender_tags,character_tags = model.execute(image)
        
        # skipped non character images
        if result == "":
            print(f"Skipped non character image: {image_path}")
            continue
        gender_subdir = "male"
        if len(gender_tags) > 0:
            for tag in gender_tags:
                if 'other' in tag:
                    gender_subdir = "other"
                if 'girl' in tag:
                    gender_subdir = "female"
        
        character_path = os.path.join(output_dir, gender_subdir)
        if len(character_tags) > 0:
            ascii_name = handle_character_name(character_tags[0])
            character_path = os.path.join(output_dir, gender_subdir, ascii_name)
            
        print(character_path)
        os.makedirs(character_path, exist_ok=True)
        text_file = os.path.join(character_path, filename + ".txt")
        
        output_image = os.path.join(character_path, filename + ".webp")
        try:
            with Image.open(image_path) as image:
                # exif = image.info['exif']
                image = ImageOps.exif_transpose(image)
                lossless, quality = (False, 90)
                image.save(output_image, 'webp', optimize = True, quality = quality, lossless = lossless)
                print("image save to ", output_image)
        except:
            print(f"Error in file {image_path}")
            os.remove(image_path)
            print(f"Removed file {image_path}")

        with open(text_file, "w", encoding="utf8") as f: 
            f.write(result)
        print("text save to ", text_file)
        