import os
import glob
from tqdm import tqdm
import torch

input_dir = "F:/ImageSet/kolors_cosplay/train"


files = glob.glob(f"{input_dir}/**", recursive=True)
image_exts = [".png",".jpg",".jpeg",".webp"]
image_files = [f for f in files if os.path.splitext(f)[-1].lower() in image_exts]

for image_file in tqdm(image_files,position=0):
    # get basename and ext
    # basename = os.path.basename(image_file)
    filename,ext = os.path.splitext(image_file)
    
    # latent file
    latent_file = image_file.replace(ext,".nplatent")
    
    # kolors file
    kolors_file = image_file.replace(ext,".npkolors")
    
    latent = torch.load(latent_file)
    # print(latent['latent'].shape)
    if torch.equal(torch.Tensor(list(latent['latent'].shape)),torch.Tensor([4, 192, 192])):
        print(latent_file)
        print(latent['latent'].shape)
        kolors = torch.load(kolors_file)
        print(kolors)
        # remove latent_file and kolors_file
        os.remove(latent_file)
        os.remove(kolors_file)
        # break