from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

model_path = r"F:\T2ITrainer_pulic\T2ITrainer\qwen_models\ocr-qwen"

dtype = torch.float16
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=dtype,
    device_map=device
)

# processor_path = r"F:\T2ITrainer_pulic\T2ITrainer\qwen_models\Qwen-Image-Edit-2509-14B\processor"
# processor = AutoProcessor.from_pretrained(processor_path)
# image_path = r"F:\ImageSet\ti2i\ti2i_test\image_0000_L.png"

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": image_path,
#             },
#             {"type": "text", "text": "Extract the text from this image."},
#         ],
#     }
# ]

# text = processor.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )
# image_inputs, video_inputs = process_vision_info(messages)
# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )
# inputs = inputs.to(device)

# generated_ids = model.generate(**inputs, max_new_tokens=512)
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print("Extracted Text:", output_text[0])
