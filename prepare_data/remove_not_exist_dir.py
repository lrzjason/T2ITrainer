import os
import shutil

txt_dir = "F:/ImageSet/kolors_pony/male_txt"
image_dir = "F:/ImageSet/kolors_pony/male"

sub_dirs = os.listdir(txt_dir)

for sub_dir in sub_dirs:
    print(sub_dir)
    txt_sub_dir_path = os.path.join(txt_dir, sub_dir)
    image_sub_dir_path = os.path.join(image_dir, sub_dir)
    txt_character_dirs = os.listdir(txt_sub_dir_path)
    image_character_dirs = os.listdir(image_sub_dir_path)
    
    difference = list(set(image_character_dirs) - set(txt_character_dirs))
    # print(difference)
    # break
    for character_dir in difference:
        full_character_path = os.path.join(image_sub_dir_path, character_dir)
        print(full_character_path)
        if os.path.exists(full_character_path):
            # remove dir
            shutil.rmtree(full_character_path)
    #     break
    # break