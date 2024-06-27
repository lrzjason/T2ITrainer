## **PyTorch should be installed above 2.3.0 and cu121**
torch>=2.3.0+cu121

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
2024-06-27: Fix incorrect bucketing on hunyuan lora training


## **Acknowledgements:**

Thanks @minienglish1 and @Freon in EveryDream discord for many help.

Thanks to kohya ss for many references from his training codebase \
Thanks to Kblueleaf for coding reference on hunyuandit gradient checkpoint implementation

