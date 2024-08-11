import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, \
                                    InterpolationMode, ToTensor, Resize, CenterCrop

from transformers import BlipForConditionalGeneration
from torchvision import transforms
import cv2
import numpy as np


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")
# model.train()
image_path = "F:/ImageSet/pixart_test_cropped/_0f810856-9473-4be9-9103-e5050ebb55a6.webp"
text = "cotton doll, A plush toy resembling a chef's attire. It has brown hair styled in a fringe, rosy cheeks with pink blushes, and black eyes. The toy is wearing a white chef hat and a white chef outfit with blue buttons on the front. Beside the toy, there's a wooden rolling pin. The setting appears to be a soft fabric surface, possibly a tablecloth, against a light-colored wall with subtle textures."

mean = [
    0.48145466,
    0.4578275,
    0.40821073
]
std = [
    0.26862954,
    0.26130258,
    0.27577711
]
try:
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is not None:
        # Convert to RGB format (assuming the original image is in BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        print(f"Failed to open {image_path}.")
except Exception as e:
    print(f"An error occurred while processing {image_path}: {e}")

text = ['a photography of' + ' ' + text]
inputs = processor(images=[image], text=text, return_tensors="pt", padding='longest')
inputs = {key: inputs[key].to("cuda") for key in inputs.keys()}
inputs['labels'] = inputs['input_ids'].masked_fill(
    inputs['input_ids'] == processor.tokenizer.pad_token_id, -100
)
prompt_length = len(processor.tokenizer('a photography of').input_ids) - 1
inputs['labels'][:, :prompt_length] = -100

with torch.autocast(device_type="cuda"):
    outputs = model(**inputs)
    reward = -outputs.loss
    
print(reward)
# score = model.score(inputs)
# print(score)
# out = model(**inputs)
# print(out)
# l_cap = out.loss
# print(l_cap)