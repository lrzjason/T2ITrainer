# import
from transformers import CLIPImageProcessor, AutoProcessor, AutoModel, AutoTokenizer


from PIL import Image
import torch
import json
import glob
import os

from tqdm import tqdm

import numpy as np
import shutil

class MPSModel():
    def __init__(self, model_name_or_path="F:/MPS/outputs/MPS_overall_checkpoint.pth", processor_name_or_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(processor_name_or_path, trust_remote_code=True)
        self.processor = CLIPImageProcessor.from_pretrained(processor_name_or_path)
        self.model = torch.load(model_name_or_path)
        self.model.eval().to(device)
        self.model.model.text_model.pad_token_id = 1
        self.model.model.text_model.bos_token_id = 49406
        self.model.model.text_model.eos_token_id = 49407
        self.condition = "light, color, clarity, tone, style, ambiance, artistry, shape, face, hair, hands, limbs, structure, instance, texture, quantity, attributes, position, number, location, word, things."
    def _process_image(self, image):
        if isinstance(image, dict):
            image = image["bytes"]
        if isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        if isinstance(image, str):
            image = Image.open( image )
        image = image.convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt")["pixel_values"]
        return pixel_values
    
    def _tokenize(self, caption):
            input_ids = self.tokenizer(
                caption,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids
            return input_ids
    def score(self, image, prompt, condition=None):
        device = self.device
        image_input = self._process_image(image).to(device)
        text_input = self._tokenize(prompt).to(device)
        clip_model = self.model
        if condition is None:
            condition = "light, color, clarity, tone, style, ambiance, artistry, shape, face, hair, hands, limbs, structure, instance, texture, quantity, attributes, position, number, location, word, things."
        condition_batch = self._tokenize(condition).repeat(text_input.shape[0],1).to(device)

        with torch.no_grad():
            text_f, text_features = clip_model.model.get_text_features(text_input)
            image_f = clip_model.model.get_image_features(image_input.half())
            condition_f, _ = clip_model.model.get_text_features(condition_batch)
            sim_text_condition = torch.einsum('b i d, b j d -> b j i', text_f, condition_f)
            sim_text_condition = torch.max(sim_text_condition, dim=1, keepdim=True)[0]
            sim_text_condition = sim_text_condition / sim_text_condition.max()
            mask = torch.where(sim_text_condition > 0.3, 0, float('-inf'))
            mask = mask.repeat(1,image_f.shape[1],1)
            image_features = clip_model.cross_model(image_f, text_f,mask.half())[:,0,:]

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_score = clip_model.logit_scale.exp() * text_features @ image_features.T
        return image_score[0]

# function return list, if single image, return list with one element. if two images, return list with two elements.
def calc_mps_multiple(images, prompt, condition, clip_model, clip_processor, tokenizer, device):
    single_infer = False
    if isinstance(images, str):
        images = [images,images]
        single_infer = True
    
    def _process_image(image):
        if isinstance(image, dict):
            image = image["bytes"]
        if isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        if isinstance(image, str):
            image = Image.open( image )
        image = image.convert("RGB")
        pixel_values = clip_processor(image, return_tensors="pt")["pixel_values"]
        return pixel_values
    
    def _tokenize(caption):
        input_ids = tokenizer(
            caption,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        return input_ids
    
    image_preprocess = []
    for image in images:
        image_preprocess.append(_process_image(image).to(device))
    
    image_inputs = torch.concatenate(image_preprocess)
    text_inputs = _tokenize(prompt).to(device)
    condition_inputs = _tokenize(condition).to(device)

    
    if single_infer:
        with torch.no_grad():
            text_features, image_0_features, image_1_features = clip_model(text_inputs, image_inputs, condition_inputs)
            image_0_features = image_0_features / image_0_features.norm(dim=-1, keepdim=True)
            # image_1_features = image_1_features / image_1_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_0_scores = clip_model.logit_scale.exp() * torch.diag(torch.einsum('bd,cd->bc', text_features, image_0_features))
            # image_1_scores = clip_model.logit_scale.exp() * torch.diag(torch.einsum('bd,cd->bc', text_features, image_1_features))
            # scores = torch.stack([image_0_scores, image_1_scores], dim=-1)
            # scores = torch.stack([image_0_scores], dim=-1)
            # probs = torch.softmax(scores, dim=-1)[0]
        return image_0_scores.cpu().tolist()
    else:
        text_features, image_0_features, image_1_features = clip_model(text_inputs, image_inputs, condition_inputs)
        image_0_features = image_0_features / image_0_features.norm(dim=-1, keepdim=True)
        image_1_features = image_1_features / image_1_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_0_scores = clip_model.logit_scale.exp() * torch.diag(torch.einsum('bd,cd->bc', text_features, image_0_features))
        image_1_scores = clip_model.logit_scale.exp() * torch.diag(torch.einsum('bd,cd->bc', text_features, image_1_features))
        # scores = torch.stack([image_0_scores, image_1_scores], dim=-1)
        # scores = torch.stack([image_0_scores], dim=-1)
        # probs = torch.softmax(scores, dim=-1)[0]
        return image_0_scores.cpu().tolist() + image_1_scores.cpu().tolist()
        

# img_0, img_1 = "F:/ImageSet/SA1B_caption_selected/others/sa_110.webp", "F:/ImageSet/SA1B_caption_selected/others/sa_111.webp"
# img_0, img_1 = "F:/ImageSet/SA1B_caption_selected/female/sa_1001.webp", "F:/ImageSet/SA1B_caption_selected/female/sa_1091.webp"
# # infer the best image for the caption
# prompt = "a city street scene with a large building, a church, and a traffic light. There is a woman walking down the street, and a car is driving by. The style of the image is black and white, giving it a classic and timeless appearance. The black and white color scheme adds a sense of nostalgia and emphasizes the architectural details of the buildings and the urban environment. The woman walking down the street and the car driving by contribute to the dynamic and lively atmosphere of the scene."

# # condition for overall
# # condition = ""

# print(calc_mps(img_0, prompt, condition, model, image_processor, tokenizer, device))

# mps_score_list = []
input_dir = "F:/ImageSet/SA1B_caption_selected"
output_dir = "F:/ImageSet/SA1B_caption_classified"
os.makedirs(output_dir,exist_ok=True)
# result_path = f"{input_dir}/mps_score.json"


# # load model
# device = "cuda"
# processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
# image_processor = CLIPImageProcessor.from_pretrained(processor_name_or_path)
# tokenizer = AutoTokenizer.from_pretrained(processor_name_or_path, trust_remote_code=True)

# model_ckpt_path = "F:/MPS/outputs/MPS_overall_checkpoint.pth"
# model = torch.load(model_ckpt_path)
# model.eval().to(device)

# model.model.text_model.pad_token_id = 1
# model.model.text_model.bos_token_id = 49406
# model.model.text_model.eos_token_id = 49407


mps_model = MPSModel()

condition = "light, color, clarity, tone, style, ambiance, artistry, shape, face, hair, hands, limbs, structure, instance, texture, quantity, attributes, position, number, location, word, things."


image_ext = ".webp"
supported_image_types = [image_ext]
files = glob.glob(f"{input_dir}/**", recursive=True)
image_files = [f for f in files if os.path.splitext(f)[-1].lower() in supported_image_types]

image_files = ["F:/ImageSet/SA1B_caption_selected/female/sa_1001.webp"]
for image_file in tqdm(image_files):
    text_file = ""
    for image_type in supported_image_types:
        if image_type in image_file:
            text_file = image_file.replace(image_ext, ".txt")
    if not os.path.exists(text_file):
        print(f"{text_file} not exists")
        continue
    
    prompt = open(text_file, "r", encoding="utf-8").read()
    mps_score = mps_model.score(image_file, prompt)
    print(mps_score)
    # mps_score_1 = calc_mps(image_file, prompt, condition, model, image_processor, tokenizer, device)[0]
    
    # print("\n")
    # print(mps_score_1)
    
    # mps_score_2 = infer_one_sample(image_file, prompt, condition, model, image_processor, tokenizer, device)
    # print(mps_score_2)

# if os.path.exists(result_path):
#     mps_score_list = json.load(open(result_path, "r", encoding='utf-8'))
# else:
    
#     # load model
#     device = "cuda"
#     processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
#     image_processor = CLIPImageProcessor.from_pretrained(processor_name_or_path)
#     tokenizer = AutoTokenizer.from_pretrained(processor_name_or_path, trust_remote_code=True)

#     model_ckpt_path = "F:/MPS/outputs/MPS_overall_checkpoint.pth"
#     model = torch.load(model_ckpt_path)
#     model.eval().to(device)

#     model.model.text_model.pad_token_id = 1
#     model.model.text_model.bos_token_id = 49406
#     model.model.text_model.eos_token_id = 49407

#     condition = "light, color, clarity, tone, style, ambiance, artistry, shape, face, hair, hands, limbs, structure, instance, texture, quantity, attributes, position, number, location, word, things."


#     image_ext = ".webp"
#     supported_image_types = [image_ext]
#     files = glob.glob(f"{input_dir}/**", recursive=True)
#     image_files = [f for f in files if os.path.splitext(f)[-1].lower() in supported_image_types]


#     for image_file in tqdm(image_files):
#         text_file = ""
#         for image_type in supported_image_types:
#             if image_type in image_file:
#                 text_file = image_file.replace(image_ext, ".txt")
#         if not os.path.exists(text_file):
#             print(f"{text_file} not exists")
#             continue
        
#         prompt = open(text_file, "r", encoding="utf-8").read()
#         mps_score = calc_mps(image_file, prompt, condition, model, image_processor, tokenizer, device)[0]
#         mps_score_list.append({
#             "image_file": image_file,
#             "text_file": text_file,
#             "prompt": prompt,
#             "mps_score": mps_score
#         })
#         # break

#     print(mps_score_list)

#     with open(result_path, "w", encoding='utf-8') as writefile:
#         writefile.write(json.dumps(mps_score_list, indent=4))
    
# scores = [item['mps_score'] for item in mps_score_list]

# # Calculate percentiles
# p25 = np.percentile(scores, 25)
# p50 = np.percentile(scores, 50)
# p75 = np.percentile(scores, 75)

# # Categorize and copy files
# for item in tqdm(mps_score_list):
#     mps_score = item['mps_score']
#     image_file = item['image_file']
#     text_file = item['text_file']

#     # Determine the category based on percentiles
#     if mps_score >= p75:
#         category = 'high'
#     elif mps_score >= p50:
#         category = 'good'
#     elif mps_score >= p25:
#         category = 'bad'
#     else:
#         category = 'worst'

#     # Create the directory if it doesn't exist
#     category_dir = os.path.join(output_dir, category)
#     os.makedirs(category_dir, exist_ok=True)

#     # Copy the image and text files to the category directory
#     shutil.copy(image_file, category_dir)
#     shutil.copy(text_file, category_dir)

#     print(f"Copied {image_file} and {text_file} to {category_dir}")
#     # break