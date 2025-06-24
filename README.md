# 🚀 T2ITrainer 
**⚠️ Development Notice:** Currently in active development - stability not guaranteed. Frequent updates - check changelogs regularly.

---

## 📅 Recent Updates
- **2025-05-03:** 🎨 Update train_flux_lora_ui_with_mask_timestep_range.py
```
train_flux_lora_ui_with_mask_timestep_range.py add a parameter reg_timestep to adjust the training objective
for example, reg_timgstep=700.
when training timestep < reg_timestep, the training objective is to learn the ground true image distrubution.
when training timestep >= reg_timestep, the training objective is to learn the factual image toward to ground true image distrubution.
```
---

## 🛡️ Prerequisites
- **PyTorch**: `torch>=2.3.0+cu121` (CUDA 12.1 supported) [![PyPI](https://img.shields.io/badge/PyTorch-2.3.0+-red)](https://pytorch.org/)

---

## 💻 Supported Training Configurations
| Model Type       | VRAM Requirements          | Status       |
|------------------|----------------------------|--------------|
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

    # NF4 Flux Fill for low gpu
    huggingface-cli download "lrzjason/flux-fill-nf4" --local-dir flux_models/fill/

    # skip if downloaded nf 4 Flux Fill
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
- **Mandatory:** `diffusers>=0.32.0`

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
- https://github.com/lrzjason/T2ITrainer/blob/main/doc/flux_fill.md

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

## Old Change logs: 
- https://github.com/lrzjason/T2ITrainer/blob/main/doc/change_logs.md

## Recent Change Logs:
- **2025-06-24:**  Added nf4 support and change download cli to nf4 version flux fill

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
- Thanks to minienglish1 and Freon in EveryDream Discord for the assistance.
- Special thanks to kohya ss for references from the training codebase.
- Thanks to Kblueleaf for coding reference on hunyuandit gradient checkpoint implementation.
- Thanks to Kolors for the open-source checkpoint.
- Thanks to comfyui for the wonderful codebase.
- Thanks to emojiiii for the setup.bat script and other updates.
- Thanks to Rohit Gandikota and related authors of Concept Sliders https://github.com/rohitgandikota/sliders
