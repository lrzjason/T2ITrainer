# code referenced from https://www.reddit.com/r/StableDiffusion/comments/1dfuicw/perturbed_sd3_experiment/

import numpy as np
import torch
import safetensors
from safetensors.torch import save_file
# import matplotlib.pyplot as plt

model_path = 'F:/models/Stable-diffusion/sd3/sd3_medium.safetensors'
model_save_path = 'F:/models/Stable-diffusion/sd3/sd3_medium_perturbed.safetensors'

model = safetensors.safe_open(model_path, 'pt')
keys = model.keys()
dic = {key:model.get_tensor(key) for key in keys}
parts = ['diffusion_model']
count = 0
for k in keys:
    if all(i in k for i in parts):
        v = dic[k]
        print(f'{k}: {v.std()}')
        dic[k] += torch.normal(torch.zeros_like(v)*v.mean(), torch.ones_like(v)*v.std()*.035)
        count += 1
print(count)
save_file(dic, model_save_path, model.metadata())