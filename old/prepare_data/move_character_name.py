import glob
from tqdm import tqdm
import os
import re
from captioner.florenceLargeFt import FlorenceLargeFtModelWrapper
import cv2
from mps.calc_mps import MPSModel
from PIL import Image

from utils.dist_utils import flush
import torch
import json

image_dir = "F:/ImageSet/kolors_pony/male"

files = glob.glob(f"{image_dir}/**", recursive=True)
image_exts = [".png",".jpg",".jpeg",".webp"]
image_files = [f for f in files if os.path.splitext(f)[-1].lower() in image_exts]

prefix = "score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, source_anime, anime, "

err_list = []
for image_file in tqdm(image_files):
    print(image_file)
    old_text_file = image_file.split("_res_")[0] + ".txt"
    text_file = os.path.splitext(image_file)[0] + ".txt"
    # print(text_file)
    # break
    # old_text_file = text_file.replace(".txt","_ori.txt")
    
    if os.path.exists(old_text_file):
        text = ""
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()
        # print(text)
        old_text = ""
        with open(old_text_file, "r", encoding="utf-8") as f:
            old_text = f.read()
        old_text = old_text.replace(prefix,"").replace("1boy, ","").replace("male focus, ","")
        
        # print(pure_prompt)
        # break
        # print(old_text)
        tags = old_text.split(",")
        character = tags[0].strip()
        series = tags[1].strip()
        character_part = f"{character}, {series}, "
        # skip incorrect ones
        if text.startswith(character_part):
            print("Error: "+image_file)
            err_list.append(image_file)
            continue
        # print(character_part)
        new_text = f"{character} from {series}, " + text.replace(character_part,"").replace("  "," ")
        print(text_file)
        print(new_text)
        # save new text to file
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(new_text)
        # print(new_text)
        # break

# save error list using json
with open("error_list.json", "w", encoding="utf-8") as f:
    json.dump(err_list, f)
