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
from pathlib import Path
import shutil

input_dir = "/root/xinglin-data/output/images/female/extra"
output_dir = "/root/xinglin-data/output/images/extra_txt"
os.makedirs(output_dir, exist_ok=True)
caption_prefix = '二次元动漫风格, anime artwork'

def remove_tag_prefix(text):
    clear_text = text.replace("1girl, ","").replace("1boy, ","").replace("1other, ","").replace("male focus, ","")
    return clear_text

def handle_replace(result):
    result = re.sub(r'A cartoon[a-zA-Z ]*?of ', '', result)
    result = re.sub(r'An animated[a-zA-Z ]*?of ', '', result)
    return result

captioner = None
# enable_caption = False
# captioner = FlorenceLargeFtModelWrapper()
mps_model = None

files = glob.glob(f"{input_dir}/**", recursive=True)
image_exts = [".png",".jpg",".jpeg",".webp"]
image_files = [f for f in files if os.path.splitext(f)[-1].lower() in image_exts and "_ori" not in f]
for image_file in tqdm(image_files):
    print(image_file)
    text_file = image_file.replace(".webp",".txt")
    if os.path.exists(text_file):
        # Extract the directory and base name
        directory = os.path.dirname(text_file)
        filename = os.path.basename(text_file)
        output_subdir = directory.replace(input_dir,output_dir)
        Path(output_subdir).mkdir(parents=True, exist_ok=True)
        print(text_file)
        print(output_subdir)
        output_text = os.path.join(output_subdir,filename)
        if not os.path.exists(output_text):
            # copy text_file to output_dir using shutil
            shutil.copy(text_file, output_text)
        # break