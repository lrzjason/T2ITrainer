import struct
import os

latent_path = r"D:\ComfyUI\output\test_00001_.latent"

def inspect_file_format(file_path):
    """Inspect the file format by reading first few bytes"""
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return

    # Read first 32 bytes to understand the format
    with open(file_path, 'rb') as f:
        header = f.read(32)
    
    print(f"File size: {os.path.getsize(file_path)} bytes")
    print(f"First 32 bytes (hex): {header.hex()}")
    print(f"First 32 bytes (raw): {repr(header)}")
    
    # Check if it starts with a common format signature
    if header.startswith(b'\x80\x02'):  # Pickle signature
        print("File appears to be in pickle format")
    elif header.startswith(b'PK'):  # ZIP signature
        print("File appears to be in ZIP format (possibly safetensors)")
    elif header.startswith(b'\x89PNG'):  # PNG signature
        print("File appears to be a PNG image")
    else:
        print("Unknown file format signature")

def try_different_loading_methods(file_path):
    """Try different methods to load the file"""
    import torch
    
    print("\nTrying torch.load with map_location...")
    try:
        latent = torch.load(file_path, map_location='cpu', weights_only=False)
        print(f"Successfully loaded with torch.load: {type(latent)}")
        if isinstance(latent, dict):
            print(f"Keys: {list(latent.keys())}")
        return latent
    except Exception as e:
        print(f"Failed with torch.load: {e}")
    
    # Try loading as safetensors if that library is available
    try:
        from safetensors import safe_open
        print("\nTrying safetensors loading...")
        with safe_open(file_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            print(f"safetensors keys: {keys}")
            for key in keys:
                tensor = f.get_tensor(key)
                print(f"  {key}: shape {tensor.shape}, dtype {tensor.dtype}")
    except ImportError:
        print("\nsafetensors library not available")
    except Exception as e:
        print(f"Failed with safetensors: {e}")
    
    return None

if __name__ == "__main__":
    inspect_file_format(latent_path)
    try_different_loading_methods(latent_path)