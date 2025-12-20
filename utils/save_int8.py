from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from diffusers import QwenImageLayeredPipeline
import torch
from diffusers import QwenImageTransformer2DModel

# Change to int8 quantization
quant_config = DiffusersBitsAndBytesConfig(
    load_in_8bit=True,  
)

model_id = r"F:\HF_Models\Qwen\qwen_image_layered"

# pipeline = QwenImageLayeredPipeline.from_pretrained(
#                 model_id,
#                 scheduler=None,
#                 vae=None,
#                 text_encoder=None,
#                 tokenizer=None,
#                 processor=None
#             )
# transformer = pipeline.transformer
transformer = QwenImageTransformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
)

# Save quantized model to reuse
transformer.save_pretrained("F:/HF_Models/Qwen/qwen_image_layered/transformer_int8")