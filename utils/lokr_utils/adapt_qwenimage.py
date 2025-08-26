"""
Example script demonstrating how to adapt LyCORIS to a QwenImageTransformer2DModel.

This script shows how to create a preset configuration tailored for the QwenImage model
and apply a LyCORIS network to it.
"""

import torch
from diffusers import QwenImageTransformer2DModel # Import the model class

# Import LyCORIS functions
from lycoris import create_lycoris, LycorisNetwork

def get_qwenimage_lycoris_preset(
    target_attn_mlp_layers=True,
    algo="lokr",  # Algorithm to use for all targeted layers
    rank=None,    # Rank/dim for LoRA/LoHa/LoKr adaptation (e.g., 64)
    alpha=None,   # Alpha for LoRA/LoHa/LoKr adaptation (e.g., 32)
    factor=None,  # Factor for LoKr/BOFT adaptation (e.g., 8) - used if rank is None
):
    """
    Generates a LyCORIS preset configuration for QwenImageTransformer2DModel.
    This version targets only specific attention and MLP layers using fnmatch patterns
    and allows specifying rank/dim and alpha directly.

    Args:
        target_attn_mlp_layers (bool): If True, configures targeting specific sub-modules.
        algo (str): Algorithm to use for the targeted layers (e.g., "lokr", "lora").
        rank (int, optional): Rank/dim for the adaptation. If provided, 'factor' is ignored for algorithms that support 'dim'.
        alpha (float, optional): Alpha scaling factor for the adaptation.
        factor (int, optional): Factor for LoKr/BOFT. Used only if 'rank' is None.

    Returns:
        dict: A preset dictionary for LycorisNetwork.apply_preset().
    """
    preset = {
        "target_module": [],  # Don't target modules by class name
        "module_algo_map": {},
        "name_algo_map": {},  # Target specific layers by name pattern
        "use_fnmatch": True,  # Enable fnmatch for name patterns
        "lora_prefix": "lycoris", # Ensure prefix matches model structure
    }

    if target_attn_mlp_layers:
        # Target the specific Linear layers using fnmatch patterns.
        preset["target_module"] = ["Linear"] # Target Linear layers to apply name matching
        
        targeted_patterns = [
            # Attention projections for image stream
            "transformer_blocks.*.attn.to_k",
            "transformer_blocks.*.attn.to_q",
            "transformer_blocks.*.attn.to_v",
            "transformer_blocks.*.attn.to_out.0",
            # Attention projections for text stream (added KV projections)
            "transformer_blocks.*.attn.add_k_proj",
            "transformer_blocks.*.attn.add_q_proj",
            "transformer_blocks.*.attn.add_v_proj",
            "transformer_blocks.*.attn.to_add_out",
            # Final Linear layers of MLPs (assuming net.2 is the final layer)
            "transformer_blocks.*.img_mlp.net.2",
            "transformer_blocks.*.txt_mlp.net.2",
        ]
        
        # Configure the algorithm parameters
        algo_config = {"algo": algo}
        if rank is not None:
            algo_config["dim"] = rank
            if alpha is not None:
                algo_config["alpha"] = alpha
        elif factor is not None:
            algo_config["factor"] = factor
        # If neither rank nor factor is provided, LyCORIS will use its defaults or potentially error.
            
        # Map all targeted patterns to the specified algorithm and parameters
        for pattern in targeted_patterns:
            preset["name_algo_map"][pattern] = algo_config
            
    return preset


def apply_lycoris_to_qwenimage(
    model, # Instance of QwenImageTransformer2DModel
    multiplier=1.0,
    preset=None,
    **kwargs # Additional kwargs for create_lycoris (e.g., linear_dim, linear_alpha, dropout)
):
    """
    Applies a LyCORIS network to a QwenImageTransformer2DModel instance.

    Args:
        model (QwenImageTransformer2DModel): The model instance to adapt.
        multiplier (float): The multiplier for the LyCORIS adaptation.
        preset (dict, optional): A preset configuration. If None, a default preset is used.
        **kwargs: Additional keyword arguments passed to `create_lycoris`.

    Returns:
        LycorisNetwork: The created and applied LyCORIS network instance.
    """
    if preset is None:
        # Use the default preset generator
        preset = get_qwenimage_lycoris_preset()
    
    # Apply the preset to the LycorisNetwork class
    # This modifies class variables, affecting subsequent network creation.
    # It's good practice to reset globals if creating multiple networks with different presets,
    # but for a single application, this is standard.
    LycorisNetwork.apply_preset(preset)
    
    # Create the LyCORIS network instance
    # `create_lycoris` takes the model, a base multiplier, and other configuration.
    # The specific `dim`, `alpha`, `algo` etc. for sub-modules are taken from the preset
    # unless overridden by kwargs (kwargs apply globally unless module-specific config exists).
    lycoris_network = create_lycoris(
        model,
        multiplier=multiplier,
        **kwargs # e.g., linear_dim=64, linear_alpha=32, dropout=0.1
    )
    
    # Apply the LyCORIS network to the model.
    # This injects the LyCORIS modules into the model's forward passes.
    lycoris_network.apply_to()
    lycoris_network.merge_to()
    return lycoris_network


# --- Example Usage ---
# This section shows how you might use the functions above in a training/inference script.
# It assumes the QwenImageTransformer2DModel is correctly registered with diffusers
# and the model weights are stored in F:\T2ITrainer_pulic\T2ITrainer\qwen_models\qwen_image_nf4.

if __name__ == "__main__":
    # --- 1. Load your QwenImageTransformer2DModel ---
    # Ensure the model class is importable and the path contains config and weights.
    # The QwenImageTransformer2DModel is located in the 'transformer' subdirectory.
    base_model_path = r"F:\T2ITrainer_pulic\T2ITrainer\qwen_models\qwen_image_nf4"
    transformer_subpath = "transformer" # Adjust if the subdirectory name is different
    model_path = f"{base_model_path}/{transformer_subpath}"
    
    print(f"Loading QwenImageTransformer2DModel from {model_path}...")
    try:
        qwenimage_model = QwenImageTransformer2DModel.from_pretrained(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Please ensure the model is correctly saved and the class is registered.")
        exit(1)

    # --- 2. Define or get a LyCORIS preset ---
    # You can use the default preset generator or define your own.
    # This example targets only specific attention and MLP layers,
    # and specifies rank (dim) and alpha directly.
    my_preset = get_qwenimage_lycoris_preset(
        target_attn_mlp_layers=True,
    )
    print("\nDefined LyCORIS preset:")
    print(my_preset)

    # --- 3. Apply LyCORIS to the model ---
    lycoris_net = apply_lycoris_to_qwenimage(
        qwenimage_model,
        multiplier=1.0,
        preset=my_preset,
        rank=16,      # Specify the rank (dim) for LoKr
        alpha=2,     # Specify the alpha for LoKr
        factor=8,   # Not used when rank is specified
        algo="lokr",  # Use LoKr for all targeted layers
        # linear_dim, linear_alpha, dropout can be specified here for global defaults
        # or rely on the preset's module-specific configurations.
    )
    print(f"\nApplied LyCORIS network with {len(lycoris_net.loras)} modules.")
    
    # --- 4. Save LyCORIS weights ---
    adapter_path = "qwenimage_lycoris_adapter.safetensors"
    print(f"\nSaving LyCORIS weights to {adapter_path}...")
    try:
        # Save with bfloat16 precision and optional metadata
        lycoris_net.save_weights(adapter_path, torch.bfloat16, metadata={"source": "QwenImage Example"})
        print("Weights saved successfully.")
    except Exception as e:
        print(f"Failed to save weights: {e}")

    # --- 5. Load LyCORIS weights into a new model instance ---
    print("\n--- Testing Loading of LyCORIS weights ---")
    # a. Load a new model instance (simulating a fresh start)
    print(f"Loading a new QwenImageTransformer2DModel from {model_path}...")
    try:
        new_qwenimage_model = QwenImageTransformer2DModel.from_pretrained(model_path)
        print("New model loaded successfully.")
    except Exception as e:
        print(f"Failed to load new model: {e}")
        exit(1)
    
    # b. Create a new LyCORIS network on the new model with the same preset
    print("Applying LyCORIS to the new model...")
    new_lycoris_net = apply_lycoris_to_qwenimage(
        new_qwenimage_model,
        multiplier=1.0,
        preset=my_preset, # Crucial: Use the exact same preset
    )
    print(f"Applied LyCORIS network to new model with {len(new_lycoris_net.loras)} modules.")
    
    # c. Load the saved weights into the new network
    print(f"Loading LyCORIS weights from {adapter_path}...")
    try:
        load_state = new_lycoris_net.load_weights(adapter_path)
        print("Weights loaded successfully.")
        print(f"Load state info: {load_state}") # Shows missing/unexpected keys if any
    except Exception as e:
        print(f"Failed to load weights: {e}")
        
    # --- 6. Verify weights (optional, for testing purposes) ---
    # Compare state dicts of original and loaded networks
    # Note: This comparison is strict and might be sensitive to device placement or minor differences.
    # A more robust check would compare specific parameter values.
    print("\nComparing state dictionaries of original and loaded networks...")
    try:
        orig_state_dict = lycoris_net.state_dict()
        loaded_state_dict = new_lycoris_net.state_dict()
        
        if orig_state_dict.keys() != loaded_state_dict.keys():
            print("Warning: State dict keys do not match exactly.")
        else:
            # Check if all parameters are close (accounting for potential fp precision issues)
            all_close = True
            for key in orig_state_dict:
                if not torch.allclose(orig_state_dict[key], loaded_state_dict[key], atol=1e-6):
                    print(f"Warning: Parameter '{key}' differs between original and loaded networks.")
                    all_close = False
                    break
            
            if all_close:
                print("Success: All parameters in state dictionaries are numerically close.")
            else:
                print("Warning: Some parameters differ. Check load state info above.")
                
    except Exception as e:
        print(f"Failed to compare state dictionaries: {e}")

    # --- 7. Use the model as normal (with loaded adapters) ---
    # The new_qwenimage_model's forward pass will now include the loaded LyCORIS adaptations.
    # During training, you would typically only train the new_lycoris_net parameters:
    # optimizer = torch.optim.AdamW(new_lycoris_net.parameters(), lr=1e-4)

    print("\nExample script execution completed.")