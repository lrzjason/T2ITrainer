import json
import os
from safetensors import safe_open
from safetensors.torch import save_file
import torch

def load_json_keys(json_path):
    """Load keys from JSON file"""
    with open(json_path, 'r') as f:
        keys = json.load(f)
    return keys

def create_key_mapping(incorrect_keys, correct_keys):
    """
    Create a mapping between incorrect keys and correct keys.
    This function attempts to create a logical mapping between the two key formats.
    """
    mapping = {}
    
    # Create a mapping based on pattern matching
    incorrect_to_correct = {}
    
    # Map based on common patterns
    for incorrect_key in incorrect_keys:
        # Try to find a corresponding correct key based on common patterns
        correct_key = find_corresponding_key(incorrect_key, correct_keys)
        if correct_key:
            mapping[incorrect_key] = correct_key
    
    return mapping

def find_corresponding_key(incorrect_key, correct_keys):
    """
    Find the corresponding correct key for an incorrect key based on pattern matching
    """
    # Remove the .scale_input or .scale_weight suffix from incorrect key to find the base
    base_key = incorrect_key
    if incorrect_key.endswith('.scale_input'):
        base_key = incorrect_key[:-12]  # Remove '.scale_input'
    elif incorrect_key.endswith('.scale_weight'):
        base_key = incorrect_key[:-13]  # Remove '.scale_weight'
    
    # Look for similar patterns in correct keys
    for correct_key in correct_keys:
        # Check if the base of the correct key matches the base from the incorrect key
        # Remove weight/bias suffixes from correct key for comparison
        correct_base = correct_key
        if correct_key.endswith('.weight'):
            correct_base = correct_key[:-7]
        elif correct_key.endswith('.bias'):
            correct_base = correct_key[:-5]
        
        # Simple pattern matching for common parts
        if base_key in correct_base or correct_base in base_key:
            # More specific matching for layer components
            if map_layer_components(base_key, correct_base):
                return correct_key
    
    # If no specific mapping found, try exact matches after normalization
    for correct_key in correct_keys:
        if normalize_key(incorrect_key) == normalize_key(correct_key):
            return correct_key
    
    return None

def map_layer_components(incorrect_key, correct_key):
    """
    Check if layer components match between keys
    """
    # This is a simplified version - in practice, you might need more sophisticated mapping
    incorrect_parts = incorrect_key.split('.')
    correct_parts = correct_key.split('.')
    
    # Check if they have similar structures
    # For example, model.layers.0.mlp.gate_proj.scale_input should map to model.language_model.layers.0.mlp.gate_proj.weight
    if len(incorrect_parts) >= 4 and len(correct_parts) >= 4:
        # Compare layer number and component
        if (incorrect_parts[1] == correct_parts[2] and 
            incorrect_parts[2] == correct_parts[3] and 
            incorrect_parts[3] == correct_parts[4]):
            return True
        # Try other common patterns
        elif is_similar_layer_structure(incorrect_parts, correct_parts):
            return True
    
    return False

def is_similar_layer_structure(incorrect_parts, correct_parts):
    """
    Check for similar layer structure patterns between keys
    """
    # Handle cases like 'model.layers.N' vs 'model.language_model.layers.N'
    if len(incorrect_parts) >= 4 and len(correct_parts) >= 5:
        # Check if it's a language model layer
        if (incorrect_parts[0] == correct_parts[0] and 
            incorrect_parts[1] == correct_parts[2] and  # layers.N vs language_model.layers.N
            incorrect_parts[2:] == correct_parts[3:]):
            return True
        
        # Check other structures
        if (incorrect_parts[0] == correct_parts[0] and
            incorrect_parts[1:] == correct_parts[2:]):
            return True
    
    return False

def normalize_key(key):
    """
    Normalize a key by removing scale_input/scale_weight and weight/bias suffixes for comparison
    """
    if key.endswith('.scale_input') or key.endswith('.scale_weight'):
        return key.rsplit('.', 1)[0]  # Remove last part
    elif key.endswith('.weight') or key.endswith('.bias'):
        return key.rsplit('.', 1)[0]  # Remove last part
    return key

def main():
    # Paths
    incorrect_json_path = "qwenvl25_7b_fp8_scaled_test.json"
    correct_json_path = "qwenvl25_7b_fp8_scaled.json"
    input_safetensors_path = "saved_model_fp8/model_fp8.safetensors"
    output_safetensors_path = "qwenvl_ocr_fp8.safetensors"
    
    print("Loading incorrect keys from:", incorrect_json_path)
    incorrect_keys = load_json_keys(incorrect_json_path)
    
    print("Loading correct keys from:", correct_json_path)
    correct_keys = load_json_keys(correct_json_path)
    
    print(f"Found {len(incorrect_keys)} incorrect keys and {len(correct_keys)} correct keys")
    
    # Create key mapping
    print("Creating key mapping...")
    key_mapping = create_key_mapping(incorrect_keys, correct_keys)
    
    print(f"Created {len(key_mapping)} key mappings")
    
    # Load the safetensors file
    print("Loading safetensors file...")
    tensors = {}
    with safe_open(input_safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    
    print(f"Loaded {len(tensors)} tensors from safetensors file")
    
    # Remap the keys
    print("Remapping keys...")
    remapped_tensors = {}
    unmapped_keys = []
    
    for original_key, tensor in tensors.items():
        if original_key in key_mapping:
            new_key = key_mapping[original_key]
            remapped_tensors[new_key] = tensor
            print(f"Remapped: {original_key} -> {new_key}")
        else:
            # If we can't find a specific mapping, try fuzzy matching
            mapped_key = find_corresponding_key(original_key, correct_keys)
            if mapped_key:
                remapped_tensors[mapped_key] = tensor
                print(f"Fuzzy remapped: {original_key} -> {mapped_key}")
            else:
                unmapped_keys.append(original_key)
                print(f"Unmapped key: {original_key}")
    
    if unmapped_keys:
        print(f"Warning: {len(unmapped_keys)} keys could not be mapped:")
        for key in unmapped_keys[:10]:  # Show first 10 unmapped keys
            print(f"  - {key}")
        if len(unmapped_keys) > 10:
            print(f"  ... and {len(unmapped_keys)-10} more")
    
    # Save the remapped tensors
    print(f"Saving remapped model to {output_safetensors_path}...")
    save_file(remapped_tensors, output_safetensors_path)
    
    print("Done! Saved remapped model as qwenvl_ocr_fp8.safetensors")

if __name__ == "__main__":
    main()