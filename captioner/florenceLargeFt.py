
from captioner.ModelWrapper import ModelWrapper
from utils.dist_utils import flush,get_device

# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoProcessor, AutoModelForCausalLM 
import torch

from PIL import Image, ImageDraw
import os
import glob
import cv2
from tqdm import tqdm

class FlorenceLargeFtModelWrapper(ModelWrapper):
    def __init__(self,device=None,dtype=None,tokenizer_repo_id=None):
        super().__init__()
        self.device = get_device(device)
        self.model_repo_id = "microsoft/Florence-2-large-ft"
        if tokenizer_repo_id == None:
            self.tokenizer_repo_id = self.model_repo_id
        if dtype == None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(self.model_repo_id, trust_remote_code=True)
        
        # self.prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
        # self.prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nWhat is shown in this image? ASSISTANT:"
        # self.prompt = "<image>Please describe the content of this image."
        self.prompt = "<MORE_DETAILED_CAPTION>"
        model = AutoModelForCausalLM.from_pretrained(self.model_repo_id, trust_remote_code=True)
        model.to(self.device)
        self.model = model
    
    def execute(self,image=None, other_prompt=""):
        model = self.model
        processor = self.processor
        # image = Image.open(requests.get(url, stream=True).raw)
        prompt = self.prompt
        if other_prompt != "":
            prompt = other_prompt
            crop_response = False

        inputs = processor(prompt, image, return_tensors="pt").to(self.device)

        # # autoregressively complete prompt
        # output = model.generate(**inputs, max_new_tokens=300)
        # response = processor.decode(output[0][2:], skip_special_tokens=True)
        # prompt = prompt.replace("<image>"," ")
        # response = response.replace(prompt,'').strip()
        
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            # do_sample=False,
            num_beams=3
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        response = processor.post_process_generation(generated_text, task=prompt, image_size=(image.shape[1], image.shape[0]))
        response = response[prompt]
        
        return response
    
# based on file size, return lossless, quality
def get_webp_params(filesize_mb):
    if filesize_mb <= 2:
        return (False, 100)
    if filesize_mb <= 4:
        return (False, 90)
    return (False, 80)
