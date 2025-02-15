# 🚀 T2ITrainer 
**⚠️ Development Notice:** Currently in active development - stability not guaranteed. Frequent updates - check changelogs regularly.

---

## 📅 Recent Updates
- **2025-02-15:** 🎨 Launched Flux Fill Training script for LoRA-based inpainting

---

## 🛡️ Prerequisites
- **PyTorch**: `torch>=2.3.0+cu121` (CUDA 12.1 supported) [![PyPI](https://img.shields.io/badge/PyTorch-2.3.0+-red)](https://pytorch.org/)

---

## 💻 Supported Training Configurations
| Model Type       | VRAM Requirements          | Status       |
|------------------|----------------------------|--------------|
| Hunyuandit 1.1   | >13GB GPU                  | ✅ Supported  |
| Kolors           | 11GB GPU                   | ✅ Supported  |
| SD3.5 (FP16 BS1) | 24GB GPU                   | ✅ Supported  |
| Flux, Flux Fill  | 24GB GPU                   | ✅ Supported  |

---

## ⚙️ Installation Guide

### 0. System Requirements
❗ **Mandatory:** Install [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) if encountering DLL errors

### 1. Automated Setup
Recommended Method
```bash
  git clone https://github.com/lrzjason/T2ITrainer.git
  cd T2ITrainer
  setup.bat
```
- Handles: Virtual Environment • Dependency Installation • Model Downloads

### 2. Manual Installation
**Clone Repository** 🌐
```bash
    git clone https://github.com/lrzjason/T2ITrainer.git
    cd T2ITrainer
```

**Virtual Environment** 🛠️
```bash
    python -m venv venv
    call venv\Scripts\activate
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Model Downloads** 📥
```bash
    # Kolors
    huggingface-cli download Kwai-Kolors/Kolors --local-dir kolors_models/

    # Flux Fill
    huggingface-cli download "black-forest-labs/FLUX.1-fill-dev" --local-dir flux_models/fill/
    
    # SD3.5 Models
    huggingface-cli download "stabilityai/stable-diffusion-3.5-large" --local-dir "sd3.5L/"
```

## 🚀 Launch Options
| Script          | Command                  | Special Notes                     |
|-----------------|--------------------------|-----------------------------------|
| Flux Fill       | `python ui_flux_fill.py` | Requires diffusers>=0.32.0, 24GB VRAM Recommended |
| Kolors          | `python ui.py`           | Needs [Fixed VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) |
| SD3.5 Large     | `python ui_sd35.py`      | 24GB VRAM Recommended            |

## 🔧 Testing & Integration
- **Kolors Workflow**:
    ```bash
        # ComfyUI Plugins
        git clone https://github.com/kijai/ComfyUI-KwaiKolorsWrapper
        git clone https://github.com/MinusZoneAI/ComfyUI-Kolors-MZ
    ```

- **Configuration Guide**: [📖 CivitAI Article](https://civitai.com/articles/7743)


## 🆘 Troubleshooting
- **Kolors Black Image Issue**: Ensure you're using [FP16 Fixed VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)
- **VRAM Limitations**: Adjust `blocks_to_swap` parameter (higher values reduce memory usage)
- **Windows DLL Errors**: Verify VC++ Redistributable installation
---

## 🔧 Parameter Configuration Guide
[![CivitAI Article](https://img.shields.io/badge/📖-Detailed_Parameter_Guide-purple)](https://civitai.com/articles/7743)

---

## 🌌 Flux Model Management

### Core Model Download
Base Model Installation
```bash
  huggingface-cli download "black-forest-labs/FLUX.1-dev" --local-dir flux_models/dev/
```

## 🎨 Flux Fill Special Requirements
Dependency Alert
    **Mandatory:** `diffusers>=0.32.0`

### Fill Model Installation
Inpainting Model Setup
```bash
  huggingface-cli download "black-forest-labs/FLUX.1-fill-dev" --local-dir flux_models/fill/
```

## ⚙️ Flux Fill Training Recommended Parameters
| Category          | Settings                      |
|-------------------|-------------------------------|
| Base Configuration| Rank 32, AdamW, Learn Rate 1e-4       |
| 24GB GPU          | 512 resolution, Batch Size 1  |
| VRAM Optimization | blocks_to_swap: 10 (15-20 for lower VRAM GPUs)  | 
| Precision         | bf16 (3090) / fp8 (40xx)      |

For more details (example dataset structure):
https://github.com/lrzjason/T2ITrainer/blob/main/doc/flux_fill.md

## 💻 VRAM Usage (bf16, blocks_to_swap=10)
<div align="center">
  <table>
    <tr>
      <td align="center">
        <strong>VRAM Peak</strong><br>
        <img src="https://github.com/lrzjason/T2ITrainer/blob/main/flux_example/fill_example_peak.png" width="400">
      </td>
      <td align="center">
        <strong>VRAM Low</strong><br>
        <img src="https://github.com/lrzjason/T2ITrainer/blob/main/flux_example/fill_example_low.png" width="400">
      </td>
    </tr>
  </table>
</div>


## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=lrzjason/T2ITrainer&type=Date)](https://star-history.com/#lrzjason/T2ITrainer&Date)

## Change Logs:
- **2024-06-17:** Initial UI release.
- **2024-06-25:** Updated hunyuandit lora training script and UI for multiple training scripts.
- **2024-06-27:** Fixed incorrect bucketing in hunyuandit lora training and added cache recreation option in UI.
- **2024-07-06:** Fixed RGB to BGR channel issue in sd3 latent.
- **2024-07-08:** Added `merge_state_dict.py` for merging Kolors and SDXL state dicts, saving as Kolors unet.
- **2024-07-11:** Added Kolors training script to UI.
- **2024-07-11:(2)** Added resume from checkpoint
- **2024-07-13:** Remove some unused dependencies
- **2024-07-13:(2)** Fix bugs on saved lora, added prefix to keys
- **2024-07-14:** Avoid loading text encoder unnecessarily after cached latent
- **2024-07-15:** Added setup.bat for windows users
- **2024-07-16:** Added save and load config.json
- **2024-07-19:** Fix model_path couldn't load the weight
- **2024-07-20:** Adjust cache logic and fix caching bug. Please recreate cache after this update.
- **2024-07-20:(2)** Add 2048 resolution, it could choose 1024, 2048 training based resolution now. It only recommanded to train 2048 resolution when you have >= 24GB vram.
- **2024-07-21:** Update debiased estimation loss and change cache resize from default to area
- **2024-07-22:** Fix when batch_size > 1, debiased estimation loss would cause error.
- **2024-07-22:(2)**  Thanks emojiiii for linux setup.sh script and related updates.
- **2024-07-28:**  Update caching logic, automatically update cache when added or deleted files.
- **2024-08-04:**  Add kolors slider training code and image generation code
- **2024-09-11:**  Merge dev branch. Fix debiased estimation loss and min-snr logic.
- **2024-09-14:**  Update captioner script: internvl2 and wd14
- **2024-09-17:**  Add more res buckets in 1024 scope
- **2024-09-30:**  T2Itrainer parameter explanation https://civitai.com/articles/7743
- **2024-09-30:(2)**  Fix small bug and remove 2048 training option.
- **2024-10-08:**  Add 512 resolution training
- **2024-10-23:**  Add sd3.5 Large training script
- **2024-10-24:**  Add freeze layers for SD3.5 training, default as 30 which would freeze 30~37 layers to keep texture
- **2024-10-30:**  Add model_path for SD3.5 training
- **2024-11-07:**  Modify sd3.5 freeze layers training
- **2025-01-02:**  Add flux lora training script. Thanks to diffusers and kohya codebase. Most code are referencied from them. (UI not ready yet. It is still in development.)
- **2025-02-15:**  Released Flux Fill Training script which could train lora for flux fill.

## Sponsor:
* Thanks to [@sourceful](https://www.sourceful.com/) support me making flux fill lora training script.

## 📬 Contact
- 𝕏 **Twitter**: [@Lrzjason](https://twitter.com/Lrzjason)
- 📧 **Email**: [lrzjason@gmail.com](mailto:lrzjason@gmail.com)
- 💬 **QQ Group**: 866612947
- 🎨 **CivitAI**: [xiaozhijason](https://civitai.com/user/xiaozhijason)


## Sponsors me for more open source projects:
<div align="center">
  <table>
    <tr>
      <td align="center">
        <p>Buy me a coffee:</p>
        <img src="https://github.com/lrzjason/Comfyui-In-Context-Lora-Utils/blob/main/image/bmc_qr.png" alt="Buy Me a Coffee QR" width="200" />
      </td>
      <td align="center">
        <p>WeChat:</p>
        <img src="https://github.com/lrzjason/Comfyui-In-Context-Lora-Utils/blob/main/image/wechat.jpg" alt="WeChat QR" width="200" />
      </td>
    </tr>
  </table>
</div>


## Acknowledgements:
- Thanks to minienglish1 and Freon in EveryDream Discord for the assistance.
- Special thanks to kohya ss for references from the training codebase.
- Thanks to Kblueleaf for coding reference on hunyuandit gradient checkpoint implementation.
- Thanks to Kolors for the open-source checkpoint.
- Thanks to comfyui for the wonderful codebase.
- Thanks to emojiiii for the setup.bat script and other updates.
- Thanks to Rohit Gandikota and related authors of Concept Sliders https://github.com/rohitgandikota/sliders
