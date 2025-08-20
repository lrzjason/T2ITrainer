from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

import torch
from diffusers import QwenImageTransformer2DModel

# Change to int8 quantization
quant_config = DiffusersBitsAndBytesConfig(
    load_in_8bit=True,  
)

model_id = r"F:\T2ITrainer\qwen_models\qwen_image"
transformer = QwenImageTransformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
)

# Save quantized model to reuse
transformer.save_pretrained("F:/T2ITrainer/qwen_models/qwen_image_int8/transformer")