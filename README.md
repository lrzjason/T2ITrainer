# 🚀 T2ITrainer 
**⚠️ Development Notice:** Currently in active development - stability not guaranteed. Frequent updates - check changelogs regularly.
---
T2ITrainer is a diffusers based training script. It aims to provide simple yet implementation for lora training.

- ❗ **Mandatory:** Update diffusers to v0.35.0 version for qwen related classes
```bash
pip install git+https://github.com/huggingface/diffusers.git
```


## 📅 Recent Updates
- **2025-08-10:**  
  - **Update**: Added new training script `train_qwen_image.py` which allows to train SOTA qwen image model
  - The training script use same layout config like `train_flux_lora_ui_kontext_new.py`
  
| Config | Usage |
|--------|-------|
| `config_qwen_single.json` | Train qwen image with a single image; leave the suffix empty to use all images without a suffix. |

- Usage: `python train_qwen_image.py --config_path config_qwen_single.json`

- **2025-08-05:**  
  - **Update**: Added new training script `train_flux_lora_ui_kontext_new.py` which allows setting the training layout in `config.json`.

| Config | Usage |
|--------|-------|
| `config_new_single.json` | Train Kontext with a single image; leave the suffix empty to use all images without a suffix. |
| `config_new_pairs.json` | Traditional Kontext training using `_T` and `_R` suffixed images. |
| `config_new_pairs_multiple.json` | Train with multiple reference images by setting suffixes like `_T`, `_R`, and `_G`. |
| `config_new_mixed.json` | Train Kontext using a mixed layout—e.g., combine traditional pair training with single-image training. |

- Usage: `python train_flux_lora_ui_kontext_new.py --config_path config_new_single.json`
- Not support UI selection yet.

## 🛡️ Prerequisites
- **PyTorch**: `torch>=2.3.0+cu121` (CUDA 12.1 supported) [![PyPI](https://img.shields.io/badge/PyTorch-2.3.0+-red)](https://pytorch.org/)

---

## 💻 Supported Training Configurations
| Model Type       | VRAM Requirements          | Status       |
|------------------|----------------------------|--------------|
| Qwen Image | 24GB GPU                   | ✅ Supported  |
| Flux Fill,Kontext| 24GB GPU                   | ✅ Supported  |
| SD3.5 | 24GB GPU                   | ✅ Supported  |
| Kolors           | 11GB GPU                   | ✅ Supported  |

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
❗ **Notice:** Only download the models you want to train. Install huggingface-cli if you haven't (or update the huggingface-cli if you have an old version).
You could find the download scripts in download_xxx.txt
```bash
    # NF4 Qwen Image
    hf download "lrzjason/qwen_image_nf4" --local-dir qwen_models/qwen_image_nf4/

    # NF4 Flux kontext
    hf download "lrzjason/flux-kontext-nf4" --local-dir flux_models/kontext/

    # NF4 Flux Fill for low gpu
    hf download "lrzjason/flux-fill-nf4" --local-dir flux_models/fill/

    # Kolors
    hf download Kwai-Kolors/Kolors --local-dir kolors_models/

    # SD3.5 Models
    hf download "stabilityai/stable-diffusion-3.5-large" --local-dir "sd3.5L/"
```

### Folder Structure
<div align="center">
  <table>
    <tr>
      <td align="center">
        <p>flux_models:</p>
        <img src="https://github.com/lrzjason/T2ITrainer/blob/qwen/doc/image/folder_structure.png" alt="flux_models" width="400" />
      </td>
      <td align="center">
        <p>kontext:</p>
        <img src="https://github.com/lrzjason/T2ITrainer/blob/qwen/doc/image/kontext.png" alt="kontext" width="400" />
      </td>
    </tr>
  </table>
</div>

## 🚀 Launch Options
| Model          | Command                  | Special Notes                     |
|-----------------|--------------------------|-----------------------------------|
| Qwen Image    | `python train_qwen_image.py` | Requires diffusers>=0.35.0dev, 24GB VRAM Recommended for nf4, 48GB VRAM Recommended for original model|
| Flux kontext    | `python ui_flux_fill.py` | Requires diffusers>=0.32.0, 24GB VRAM Recommended |
| Flux Fill       | `python ui_flux_fill.py` | Requires diffusers>=0.32.0, 24GB VRAM Recommended |
| Kolors          | `python ui.py`           | Needs [Fixed VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) |
| SD3.5 Large     | `python ui_sd35.py`      | 24GB VRAM Recommended            |

## 🔧 Parameter Configuration Guide
[![CivitAI Article](https://img.shields.io/badge/📖-Detailed_Parameter_Guide-purple)](https://civitai.com/articles/7743)

---
## 🌌 Qwen Model Management

### Qwen Model Installation
Inpainting Model Setup
```bash
  hf download"lrzjason/qwen_image_nf4" --local-dir qwen_models/qwen_image_nf4/
```
For more details (example dataset):
- https://github.com/lrzjason/T2ITrainer/blob/qwen/doc/qwen.md

## ⚙️ Qwen Recommended Parameters
## NF4
| Category          | Settings                      |
|-------------------|-------------------------------|
| Base Configuration| Rank 32, AdamW, Learn Rate 1e-4       |
| 24GB GPU          | 512 resolution, Batch Size 1  |
| Precision         | bf16      |

## Original Model
| Category          | Settings                      |
|-------------------|-------------------------------|
| Base Configuration| Rank 32~64, AdamW, Learn Rate 1e-4       |
| 48GB GPU          | 1024 resolution, Batch Size 1  |
| Precision         | bf16      |


## 💻 VRAM Usage (nf4, bs1,  blocks_to_swap=20)
<div align="center">
  <table>
    <tr>
      <td align="center">
        <strong>VRAM Peak</strong><br>
        <img src="https://github.com/lrzjason/T2ITrainer/blob/qwen/doc/image/qwen_nf4_block_swap_20.png" width="400">
      </td>
    </tr>
  </table>
</div>

## 💻 VRAM Usage (nf4, bs1,  blocks_to_swap=0)
<div align="center">
  <table>
    <tr>
      <td align="center">
        <strong>VRAM Peak</strong><br>
        <img src="https://github.com/lrzjason/T2ITrainer/blob/qwen/doc/image/qwen_nf4_block_swap_0.png" width="400">
      </td>
    </tr>
  </table>
</div>


## 💻 VRAM Usage (Original, bf16, bs1, blocks_to_swap=0)
<div align="center">
  <table>
    <tr>
      <td align="center">
        <strong>VRAM Peak</strong><br>
        <strong>Around 43GB</strong>
      </td>
    </tr>
  </table>
</div>

## 🌌 Flux Model Management

### Kontext Model Installation
Inpainting Model Setup
```bash
  hf download"lrzjason/flux-kontext-nf4" --local-dir flux_models/kontext/
```
For more details (example dataset):
- https://github.com/lrzjason/T2ITrainer/blob/qwen/doc/image/flux_kontext.md
- https://huggingface.co/datasets/lrzjason/object_removal_alpha_kontext

### Fill Model Installation (Skip if train kontext)
Inpainting Model Setup
```bash
  hf download"lrzjason/flux-fill-nf4" --local-dir flux_models/fill/ 
```
For more details (example dataset):
- https://github.com/lrzjason/T2ITrainer/blob/qwen/doc/image/flux_fill.md
- https://huggingface.co/datasets/lrzjason/ObjectRemovalAlpha

### Dev Model Download (Skip if train fill and kontext)
Dev Model Installation
```bash
  hf download"black-forest-labs/FLUX.1-dev" --local-dir flux_models/dev/
```

## ⚙️ Flux Training Recommended Parameters
| Category          | Settings                      |
|-------------------|-------------------------------|
| Base Configuration| Rank 16, AdamW, Lr 1e-4       |
| 24GB GPU          | 512 resolution, Batch Size 1  |
| VRAM Optimization | Use nf4 based training        | 
| Precision         | bf16     |

## 💻 VRAM Usage nf4
<div align="center">
  <table>
    <tr>
      <td align="center">
        <strong>VRAM Peak</strong><br>
        <img src="https://github.com/lrzjason/T2ITrainer/blob/qwen/doc/image/nf4_example.png" width="400">
      </td>
    </tr>
  </table>
</div>

## 💻 VRAM Usage (bf16, blocks_to_swap=10)
<div align="center">
  <table>
    <tr>
      <td align="center">
        <strong>VRAM Peak</strong><br>
        <img src="https://github.com/lrzjason/T2ITrainer/blob/qwen/doc/image/fill_example_peak.png" width="400">
      </td>
      <td align="center">
        <strong>VRAM Low</strong><br>
        <img src="https://github.com/lrzjason/T2ITrainer/blob/qwen/doc/image/fill_example_low.png" width="400">
      </td>
    </tr>
  </table>
</div>

## 🔧 Visualize Training Data
Register WandB before using it
Setup WandB env
```bash
pip install wandb
wandb login
```

Install Tensorboard first if you choice to use Tensorboard
To visualize training data, run the following command in your terminal:
```bash
tensorboard --logdir=.\logs
```

## 🔧 Kolors Testing & Integration
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


## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=lrzjason/T2ITrainer&type=Date)](https://star-history.com/#lrzjason/T2ITrainer&Date)

## Old Change logs: 
- https://github.com/lrzjason/T2ITrainer/blob/qwen/doc/change_logs.md

## Recent Change Logs:
- **2025-07-30:** 
- **Fix**: Remove text attention mask in lora training.

## Sponsor:
* Thanks to [@sourceful](https://www.sourceful.com/) support me making flux fill lora training script.

## 📬 Contact
- 𝕏 **Twitter**: [@Lrzjason](https://twitter.com/Lrzjason)
- 📧 **Email**: [lrzjason@gmail.com](mailto:lrzjason@gmail.com)
- 💬 **QQ Group**: 866612947
- 💬 **WeChat ID**: fkdeai
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
- Thanks to chenpipi0807 for Chinese translation and language switch support
- Thanks for diffusers and Terminus Research Group
- Thanks to minienglish1 and Freon in EveryDream Discord for the assistance.
- Special thanks to kohya ss for references from the training codebase.
- Thanks to Kblueleaf for coding reference on hunyuandit gradient checkpoint implementation.
- Thanks to Kolors for the open-source checkpoint.
- Thanks to comfyui for the wonderful codebase.
- Thanks to emojiiii for the setup.bat script and other updates.
- Thanks to Rohit Gandikota and related authors of Concept Sliders https://github.com/rohitgandikota/sliders