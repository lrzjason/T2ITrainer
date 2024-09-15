from diffusers import (
    AutoencoderKL,
)
from diffusers.image_processor import VaeImageProcessor
import torch
from PIL import Image
import numpy

from torchvision import transforms
import cv2
import torchvision.transforms as T
import torchvision
import glob
import os
import gc

input_dir = "F:/ImageSet/kolors_cosplay/train"

files = glob.glob(f"{input_dir}/**", recursive=True)
npkolors_exts = [".npkolors"]
npkolors_files = [f for f in files if os.path.splitext(f)[-1].lower() in npkolors_exts]

missing_time_id = []

for file in npkolors_files:
    latents = torch.load(file)
    print(latents)
    if not 'time_id' in latents:
        missing_time_id.append(file)
        try:
            os.remove(file)
        except:
            print("remove failed")


gc.collect()
torch.cuda.empty_cache()
        
print(missing_time_id)
for file in missing_time_id:
    try:
        if os.path.exists(file):
            os.remove(file)
    except:
        print("remove failed")

