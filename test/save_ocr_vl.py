from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import safetensors
import os

model_path = r"F:\T2ITrainer_pulic\T2ITrainer\qwen_models\ocr-qwen"

dtype = torch.float16
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=dtype,
    device_map=device
)
processor_path = r"F:\T2ITrainer_pulic\T2ITrainer\qwen_models\Qwen-Image-Edit-2509-14B\processor"
processor = AutoProcessor.from_pretrained(processor_path)

# Function to save the model as a single file using safetensors
def save_model_as_single_file(model, processor, output_dir="./saved_model_single_file"):
    """
    Save the model and processor as a single file using safetensors format
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model state dict
    state_dict = model.state_dict()
    
    # Convert to the format required by safetensors
    # Safetensors expects PyTorch tensors
    state_dict_torch = {}
    for key, value in state_dict.items():
        # Convert to CPU
        state_dict_torch[key] = value.cpu().detach()
        
    # Save using safetensors
    from safetensors.torch import save_file
    model_path = os.path.join(output_dir, "model.safetensors")
    save_file(state_dict_torch, model_path)
    
    # Save processor separately (it contains tokenizer and config)
    processor.save_pretrained(output_dir)
    
    print(f"Model saved as single file: {model_path}")
    print(f"Processor saved to: {output_dir}")

# Alternative implementation: Save with specific quantization (FP8 if supported)
def save_model_as_single_file_fp8(model, processor, output_dir="./saved_model_fp8"):
    """
    Save the model as a single file using FP8 precision (if supported)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert model weights to FP8 if possible
    # Note: Full FP8 support might require special handling and may not be universally supported
    state_dict = model.state_dict()
    
    # For FP8, we need to convert the weights appropriately
    # This is a simplified approach - actual FP8 implementation might require more sophisticated methods
    state_dict_fp8 = {}
    
    for key, value in state_dict.items():
        # Convert to float8 (Note: torch.float8 is not yet natively supported in most PyTorch versions)
        # Using e4m3fn or e5m2 formats if available
        if hasattr(torch, 'float8_e4m3fn'):
            # Try to convert to float8 format if available
            try:
                converted_value = value.to(torch.float8_e4m3fn if value.numel() > 0 else torch.float8_e5m2)
                state_dict_fp8[key] = converted_value.cpu().detach()
            except:
                # Fallback to float16 if float8 conversion fails
                state_dict_fp8[key] = value.half().cpu().detach()
        else:
            # If float8 is not available, save as float16 as the next best option for size reduction
            state_dict_fp8[key] = value.half().cpu().detach()
    
    # Save using safetensors
    from safetensors.torch import save_file
    model_path = os.path.join(output_dir, "model_fp8.safetensors")
    save_file(state_dict_fp8, model_path)
    
    # Save processor separately
    processor.save_pretrained(output_dir)
    
    print(f"Model saved as single file with reduced precision: {model_path}")
    print(f"Processor saved to: {output_dir}")

# Save the model as a single file
print("Saving model as single file...")
# save_model_as_single_file(model, processor, "./saved_model_single_file")

print("\nSaving model as single file with FP8 (or FP16 fallback)...")
save_model_as_single_file_fp8(model, processor, "./saved_model_fp8")

print("\nModel saved successfully!")