
from compel import Compel, ReturnedEmbeddingsType
import torch
from diffusers import StableDiffusionXLPipeline

device = "cuda"
model_path = "F:/models/Stable-diffusion/sdxl/tPonynai3_v6.safetensors"
weight_dtype = torch.float16
pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,variant="fp16", use_safetensors=True, 
    torch_dtype=torch.float16).to("cuda")

pipe.unet.to(device, dtype=weight_dtype)
compel = Compel(tokenizer=[pipe.tokenizer, pipe.tokenizer_2] , text_encoder=[pipe.text_encoder, pipe.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
# scheduler = DPMSolverMultistepScheduler(
#     beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, 
#     use_karras_sigmas=True,algorithm_type='dpmsolver++',solver_order=2
# )

# compel = Compel(tokenizer=[pipe.tokenizer, pipe.tokenizer_2] , text_encoder=[pipe.text_encoder, pipe.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
neg_prompt = "score_6, score_5, score_4,  source_pony, source_furry, deformed, bad anatomy, disfigured, poorly drawn face, watermark, web adress, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, disconnected head, malformed hands, long neck, mutated hands and fingers, bad hands, missing fingers, worst quality, low quality, mutation, poorly drawn, "
prompt_embeds, pooled_prompt_embeds = compel(neg_prompt)

print(neg_prompt)