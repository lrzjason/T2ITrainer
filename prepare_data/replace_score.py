import glob
from tqdm import tqdm
import os

output_dir = "F:/ImageSet/kolors_pony/female/blue_archive"

files = glob.glob(f"{output_dir}/**", recursive=True)
image_exts = [".png",".jpg",".jpeg",".webp"]
image_files = [f for f in files if os.path.splitext(f)[-1].lower() in image_exts and "_ori" not in f]
for image_file in tqdm(image_files):
    text_file = os.path.splitext(image_file)[0] + ".txt"
    
    print(text_file)
    # read text file
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    if "score" in text:
        print(text)
        text = text.replace(",  score_10_down","").replace(", score_10_up","")
        text = text.replace(", score_15_up","")
        text = text.strip()
        print(text)
        
    # write actual prompt to text path
    with open(text_file, "w", encoding='utf-8') as writefile:
        # save file
        writefile.write(text)