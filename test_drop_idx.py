from transformers import Qwen2Tokenizer, Qwen2VLProcessor, Qwen2_5_VLForConditionalGeneration
import torch


device = "cuda"
pretrained_model_name_or_path = r"F:\T2ITrainer_pulic\T2ITrainer\qwen_models\qwen_image_edit_plus_nf4"
# Offload models to CPU and load necessary components
tokenizer = Qwen2Tokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
)

processor = Qwen2VLProcessor.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="processor",
)


text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder"
).to(device)



max_sequence_length = 1024
llm_template = None
prompt = "a cat behind a wall"
prompt = [prompt] if isinstance(prompt, str) else prompt
template_start = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n"
template = template_start + "{}<|im_end|>\n<|im_start|>assistant\n"
drop_idx = 34
if llm_template is not None:
    template = llm_template
txt = [template.format(e) for e in prompt]

model_inputs = tokenizer(
    template_start, max_length=max_sequence_length + drop_idx, padding=True, truncation=True, return_tensors="pt"
).to(device)

model_inputs = tokenizer(
    txt, max_length=max_sequence_length + drop_idx, padding=True, truncation=True, return_tensors="pt"
).to(device)


outputs = text_encoder(
    input_ids=model_inputs.input_ids,
    attention_mask=model_inputs.attention_mask,
    output_hidden_states=True,
)


def extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

    return split_result

hidden_states = outputs.hidden_states[-1]
split_hidden_states = extract_masked_hidden(hidden_states, model_inputs.attention_mask)
split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
prompt_embed_length = max([e.size(0) for e in split_hidden_states])
prompt_embeds = torch.stack(
    [torch.cat([u, u.new_zeros(prompt_embed_length - u.size(0), u.size(1))]) for u in split_hidden_states]
)
encoder_attention_mask = torch.stack(
    [torch.cat([u, u.new_zeros(prompt_embed_length - u.size(0))]) for u in attn_mask_list]
)

# Decode the embeddings to text
# Using the text encoder's decoder to convert embeddings back to token IDs
with torch.no_grad():
    # Get the decoder part of the text encoder
    decoder = text_encoder.get_decoder()
    
    # Decode embeddings using the decoder
    decoded_logits = decoder(
        inputs_embeds=prompt_embeds,
        attention_mask=encoder_attention_mask
    ).last_hidden_state
    
    # Get the output projection layer to convert to vocabulary space
    lm_head = text_encoder.lm_head
    
    # Apply the language modeling head to get logits
    logits = lm_head(decoded_logits)
    
    # Get the most likely tokens
    decoded_tokens = torch.argmax(logits, dim=-1)
    
# Convert the decoded tokens to text
decoded_texts = tokenizer.batch_decode(decoded_tokens, skip_special_tokens=True)
print("Decoded texts:")
for i, text in enumerate(decoded_texts):
    print(f"[{i}]: {text}")
