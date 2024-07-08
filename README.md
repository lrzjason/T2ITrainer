## **PyTorch should be installed above 2.3.0 and cu121**
torch>=2.3.0+cu121

## For Kolors merging SDXL checkpoint
## **!!!Important!!!** 
After a discussion of the merging effective, it might not have very meaningful of merging kolors with SDXL checkpoint.
It does affected the weight but it might not able to transfer the knowledge of SDXL checkpoint to Kolors.
It more like adding randoms noise the Kolors and reverse some overfitting from the Kolors training.
## **Hardware requirements:** 
Requires 20GB+ RAM for model merging

Please activate your virtual environment if needed.
## **1. Install dependencies:**
```
git clone https://github.com/lrzjason/T2ITrainer.git

cd T2ITrainer

pip install -r requirements.txt
```
## **2. Run the script:**
```
python ui_for_merge.py
```

## **3. Testing**
At 2024-07-08, Kolors inference is very limited. \
I am using this repo for the testing https://github.com/kijai/ComfyUI-KwaiKolorsWrapper \
Backup the original Kolors model first and replace the merged verion to ComfyUI\models\diffusers\Kolors\unet\diffusion_pytorch_model.fp16.safetensors

## For hunyuan dit 1.1 training
## **Hardware requirements:** 
Requires a cuda gpu above 13GB of memory for hunyuan lora training.

Please activate your virtual environment if needed.
## **1. Install dependencies:**
```
git clone https://github.com/lrzjason/T2ITrainer.git

cd T2ITrainer

pip install -r requirements.txt
```
## **2. Run the script:**
```
python ui.py
```
## **3. HunyuanDit lora testing:**
Modify the test.py output_dir, lora_dir_name and prompt to generate image with lora.
```
python test.py
```

## **Change logs:**
2024-06-17: UI Initial release \
2024-06-25: Update hunyuan lora training script, update UI for multiple training script \
2024-06-27: Fix incorrect bucketing on hunyuan lora training, add recreate cache option in ui
2024-07-06: Fix wrong channel, RGB to BGR in sd3 latent
2024-07-08: Added a merge_state_dict.py for merging kolors and normal sdxl state dict, save as kolors unet


## **Acknowledgements:**

Thanks @minienglish1 and @Freon in EveryDream discord for many help.

Thanks kohya ss for many references from his training codebase \
Thanks Kblueleaf for coding reference on hunyuandit gradient checkpoint implementation
Thanks Kolors for the open source checkpoint
Thanks comfyui for his wonderful codebase
