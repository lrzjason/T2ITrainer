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

files = glob.glob(f"{output_dir}/**", recursive=True)
image_exts = [".png",".jpg",".jpeg",".webp"]
image_files = [f for f in files if os.path.splitext(f)[-1].lower() in image_exts and "_ori" not in f]

for image_file in tqdm(image_files):
    text_file = os.path.splitext(image_file)[0] + ".txt"
    old_text_file = text_file.replace(".txt","_ori.txt")
    
    text = open(text_file, "r", encoding="utf-8").read()
    # skip processed
    if "score_" in text:
        continue
    
    old_text = open(old_text_file, "r", encoding="utf-8").read()
    score_tag = old_text.split(",")[-1].strip()
    print(old_text_file)
    print(score_tag)
    if "score_" not in score_tag:
        print("not score tag, skipped")
        continue
    
    # add score_tag to text end and save to text file
    text += f", {score_tag}"
    text = text.replace(",,",",")
    
    with open(text_file, "w", encoding='utf-8') as writefile:
        # save file
        writefile.write(text)
    print(text_file)

    # break
    