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

output_dir = "F:/ImageSet/kolors_pony/female/blue_archive"
caption_prefix = '二次元动漫风格, anime artwork'

files = glob.glob(f"{output_dir}/**", recursive=True)
image_exts = [".png",".jpg",".jpeg",".webp"]
image_files = [f for f in files if os.path.splitext(f)[-1].lower() in image_exts and "_ori" not in f]

for image_file in tqdm(image_files):
    text_file = os.path.splitext(image_file)[0] + ".txt"
    old_text_file = text_file.replace(".txt","_ori.txt")
    
    if os.path.exists(text_file):
        text = open(text_file, encoding="utf-8").read()
        if caption_prefix not in text:
            if os.path.exists(old_text_file):
                # read old text
                old_text = open(old_text_file, encoding="utf-8").read()
                
                with open(text_file, "w", encoding='utf-8') as writefile:
                    # save file
                    writefile.write(old_text)
                print("\n")
                print(text_file)
                print(text)
                print(old_text)
                # break