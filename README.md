# T2ITrainer README

## Prerequisites
- PyTorch version >= 2.3.0 with CUDA 12.1 support (`torch>=2.3.0+cu121`).

## Supported Training Configurations
- For Lora training: Supports kolors, sd3, and hunyuandit 1.1.
  - **Hardware requirements:**
    - CUDA GPU with >13GB memory for hunyuandit lora training.
    - CUDA GPU with >24GB memory for kolors training.

Please activate your virtual environment if needed.

## Installation

### 1. Install dependencies:
```
git clone https://github.com/lrzjason/T2ITrainer.git
cd T2ITrainer
pip install -r requirements.txt
```

### 2. Run the script:
```
python ui.py
```

### 3. Testing:
- For hunyuandit 1.1:
  Modify `test.py` with `output_dir`, `lora_dir_name`, and prompt to generate images with lora.
  ```
  python test.py
  ```

- For kolors:
  Install plugins from:
  - [ComfyUI-KwaiKolorsWrapper](https://github.com/kijai/ComfyUI-KwaiKolorsWrapper)
  - [ComfyUI-Kolors-MZ](https://github.com/MinusZoneAI/ComfyUI-Kolors-MZ)
  
  Then use the normal lora loader in ComfyUI.

## Kolors Merging SDXL Checkpoint

### **!!! Important !!!**
After merging, note that it may not effectively transfer knowledge from SDXL checkpoint to Kolors. It might add random noise and reverse some overfitting.

- **Hardware requirements:**
  Requires 20GB+ RAM for model merging.

Please activate your virtual environment if needed.

### 1. Install dependencies:
```
git clone https://github.com/lrzjason/T2ITrainer.git
cd T2ITrainer
pip install -r requirements.txt
```

### 2. Run the script:
```
python ui_for_merge.py
```

### 3. Testing:
As of 2024-07-08, Kolors inference is limited. Refer to [ComfyUI-KwaiKolorsWrapper](https://github.com/kijai/ComfyUI-KwaiKolorsWrapper) for testing.

## Change Logs:
- **2024-06-17:** Initial UI release.
- **2024-06-25:** Updated hunyuandit lora training script and UI for multiple training scripts.
- **2024-06-27:** Fixed incorrect bucketing in hunyuandit lora training and added cache recreation option in UI.
- **2024-07-06:** Fixed RGB to BGR channel issue in sd3 latent.
- **2024-07-08:** Added `merge_state_dict.py` for merging Kolors and SDXL state dicts, saving as Kolors unet.
- **2024-07-11:** Added Kolors training script to UI.

## Acknowledgements:
- Thanks to @minienglish1 and @Freon in EveryDream Discord for their assistance.
- Special thanks to kohya ss for references from their training codebase.
- Thanks to Kblueleaf for coding reference on hunyuandit gradient checkpoint implementation.
- Thanks to Kolors for the open-source checkpoint.
- Thanks to comfyui for the wonderful codebase.
