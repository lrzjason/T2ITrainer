import os
import glob
import random
from tqdm import tqdm

input_dir = "F:/ImageSet/SA1B_caption"

output_dir = "F:/ImageSet/SA1B_caption_random"
os.makedirs(output_dir,exist_ok=True)

male_output_dir = f"{output_dir}/male"
os.makedirs(male_output_dir,exist_ok=True)
female_output_dir = f"{output_dir}/female"
os.makedirs(female_output_dir,exist_ok=True)
others_output_dir = f"{output_dir}/others"
os.makedirs(others_output_dir,exist_ok=True)

supported_image_types = ['.txt']
files = glob.glob(f"{input_dir}/**", recursive=True)
text_files = [f for f in files if os.path.splitext(f)[-1].lower() in supported_image_types]

random.shuffle(text_files)

# male_prompts = []
male_count = 0
# female_prompts = []
female_count = 0
# others_prompts = []
others_count = 0

total_count = 10

for file in tqdm(text_files):
    print(file)
    with open(file, 'r', encoding="utf-8") as f:
        # read file all text 
        contents = f.read() 
        contents = contents.replace('\n', ' ').strip()
        if male_count < total_count and (" male " in contents.lower() or " man "  in contents.lower() or " boy "  in contents.lower() or " he "  in contents.lower() or " his "  in contents.lower()):
            # male_prompts.append(contents)
            male_count += 1
            print("male: ", male_count)
            # save the content to output file
            with open(f"{male_output_dir}/{os.path.basename(file)}",'w', encoding="utf-8") as f:
                f.write(contents)
        elif female_count < total_count and (" female " in contents.lower() or " woman "  in contents.lower() or " girl "  in contents.lower() or " she "  in contents.lower() or " her "  in contents.lower()):
            # female_prompts.append(contents)
            female_count += 1 
            print("female: ", female_count)
            # save the content to output file
            with open(f"{female_output_dir}/{os.path.basename(file)}",'w', encoding="utf-8") as f:
                f.write(contents)
        elif others_count < total_count:
            # others_prompts.append(contents)
            others_count += 1 
            print("others: ", others_count)
            # save the content to output file
            with open(f"{others_output_dir}/{os.path.basename(file)}",'w', encoding="utf-8") as f:
                f.write(contents)
        
        if male_count >= total_count and female_count >= total_count and others_count >= total_count:
            break
            
        # break