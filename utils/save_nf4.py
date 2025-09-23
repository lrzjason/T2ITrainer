from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig


import torch
from diffusers import QwenImageTransformer2DModel

quant_config = DiffusersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = r"F:\T2ITrainer\qwen_models\qwen_image_edit_plus"
transformer = QwenImageTransformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
)

# save quantized model to reuse
transformer.save_pretrained(r"F:\T2ITrainer\qwen_models\qwen_image_edit_plus_nf4")
