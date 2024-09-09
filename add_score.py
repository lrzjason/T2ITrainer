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

output_dir = "F:/ImageSet/kolors_pony/female/one_piece"
caption_prefix = '二次元动漫风格, anime artwork'

def handle_replace(result):
    result = re.sub(r'A cartoon[a-zA-Z ]*?of ', '', result)
    result = re.sub(r'An animated[a-zA-Z ]*?of ', '', result)
    return result

captioner = None
enable_caption = True
# captioner = FlorenceLargeFtModelWrapper()
mps_model = None

files = glob.glob(f"{output_dir}/**", recursive=True)
image_exts = [".png",".jpg",".jpeg",".webp"]
image_files = [f for f in files if os.path.splitext(f)[-1].lower() in image_exts and "_ori" not in f]
with torch.no_grad():
    for image_file in tqdm(image_files):
        text_file = os.path.splitext(image_file)[0] + ".txt"
        old_text_file = text_file.replace(".txt","_ori.txt")
        print(text_file)
        if os.path.exists(old_text_file) and enable_caption:
            text = open(text_file, "r", encoding="utf-8").read()
            # skip processed
            if caption_prefix in text:
                continue
            image = cv2.imread(image_file)
            if captioner is None:
                captioner = FlorenceLargeFtModelWrapper()
            result = captioner.execute(image)

            result = handle_replace(result)

            new_content = f"{caption_prefix}, {result} {text}"
            print(new_content)
            
            print("\n",text_file)
            print(text)
            with open(text_file, "w", encoding='utf-8') as writefile:
                # save file
                writefile.write(new_content)
            del image
            flush()
        else:
            with open(text_file, "r", encoding="utf-8") as f:
                text = f.read()
            
            if "score_" not in text:
                # image = cv2.imread(image_file)
                image = Image.open(image_file)
                if mps_model is None:
                    mps_model = MPSModel()
                # print("\n",text)
                score = mps_model.score(image,text).item()
                # print(score)
                # avoid double score
                score_prompt = "below_score_2"
                if score >= 2 and score < 5:
                    score_prompt = "below_score_5"
                elif score >= 5 and score < 10:
                    score_prompt = "below_score_10"
                elif score >= 13:
                    score_prompt = "score_13_up"
                elif score >= 15:
                    score_prompt = "score_15_up"
                elif score >= 10:
                    score_prompt = "score_10_up"
                text += f", {score_prompt}"
                # print(text)
                # break
                # write actual prompt to text path
                # print("\n",old_text_file)
                # print(text)
                # if os.path.exists(old_text_file):
                #     continue
                with open(old_text_file, "w", encoding='utf-8') as writefile:
                    # save file
                    writefile.write(text)

        
            # if os.path.exists(old_text_file):
            #     text = open(old_text_file, "r", encoding="utf-8").read()
            
            # image = cv2.imread(image_file)
            # result = captioner.execute(image)

            # result = handle_replace(result)

            # new_content = f"{caption_prefix}, {result} {text}"
            # print(new_content)
            
            # print("\n",text_file)
            # print(text)
            # with open(text_file, "w", encoding='utf-8') as writefile:
            #     # save file
            #     writefile.write(new_content)
            # del image
            # flush()
            # break