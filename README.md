# 🚀 T2ITrainer 
**⚠️ Development Notice:** Currently in active development - stability not guaranteed. Frequent updates - check changelogs regularly.
---
T2ITrainer is a diffusers based training script. It aims to provide simple yet implementation for lora training.

- ❗ **Mandatory:** Update diffusers to latest github version

```bash
pip install git+https://github.com/huggingface/diffusers.git -U
```


## 📅 Major Updates
- **2025-12-26:** Support 2511 training
- **2025-12-24:**  New Backend Architecture with SQLite-based job queue system. Replaced Redis with SQLite for better reliability and simplified deployment. Added API Service, Worker Service, and Streamer Service for improved training job management and real-time output streaming.
- **2025-12-23:**  Enhanced Training Log System with persistent logging, status tracking, and retrieval capabilities. Training logs are now stored separately by date with JSON format, allowing retrieval even when the web page is closed. Training status is persisted to maintain state across browser sessions.
- **2025-12-20:**  Node Based Frontend UI for configuration with visualization capabilities. Flexible dataset configuration. (Still under development)
- **2025-12-20:**  Support LongCat Image and LongCat Edit, 6B MMDIT flux vae models, Lora Training

## 🛡️ Prerequisites
- **PyTorch**: `torch>=2.3.0+cu121` (CUDA 12.1 supported) [![PyPI](https://img.shields.io/badge/PyTorch-2.3.0+-red)](https://pytorch.org/)
- **Node.js**: `node>=14.0.0` (Required for frontend UI) [![Node.js](https://img.shields.io/badge/Node.js-14.0.0+-green)](https://nodejs.org/)

---

## 💻 Supported Training Configurations
| Model Type       | VRAM Requirements          | Status       |
|------------------|----------------------------|--------------|
| LongCat Image/Edit | 24GB GPU | ✅ Supported  |
| Qwen Edit | 48GB GPU (bf16)| ✅ Supported  |
| Qwen Image | 24GB GPU (nf4) 48GB GPU (bf16)| ✅ Supported  |
| Flux Fill, Kontext| 24GB GPU                   | ✅ Supported  |

---

## ⚙️ Installation Guide

### 0. System Requirements
❗ **Mandatory:** Install [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) if encountering DLL errors

### 0.1 Frontend Requirements
❗ **Mandatory:** Install [Node.js](https://nodejs.org/) (version 14 or higher) for the Node-Based Frontend UI

After installing Node.js, verify the installation:
```bash
node --version
npm --version
```

```bash
    cd frontend
    npm install
    npm run build
    cd ..
```

### 1. Automated Setup
Recommended Method
```bash
  git clone https://github.com/lrzjason/T2ITrainer.git
  cd T2ITrainer
  setup.bat
```
- Handles: Virtual Environment • Dependency Installation • Model Downloads • Frontend Dependencies

The automated setup will:
1. Create a Python virtual environment
2. Install Python dependencies
3. Install Node.js dependencies for the frontend
4. Build the frontend UI
5. Download required models

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

**Frontend Setup** 🖥️
```bash
    cd frontend
    npm install
    npm run build
    cd ..
```

**Backend Dependencies** 📦
```bash
    pip install -r requirements.txt
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

    # download original repo for lokr training
    hf download "Qwen/Qwen-Image" --local-dir qwen_models/qwen_image/
    hf download "Qwen/Qwen-Image-Edit" --local-dir qwen_models/qwen_image_edit/
```

## 🚀 Launch Options

### Command Line Interface
| Model          | Command                  | Special Notes                     |
|-----------------|--------------------------|-----------------------------------|
| Qwen Edit    | `python train_qwen_image_edit.py` | 48GB VRAM Recommended for original model|
| Qwen Image    | `python train_qwen_image.py` | 24GB VRAM Recommended for nf4, 48GB VRAM Recommended for original model|
| Flux kontext    | `python ui_flux_fill.py` | 24GB VRAM Recommended |
| Flux Fill       | `python ui_flux_fill.py` | 24GB VRAM Recommended |
| LongCat Image   | `python train_longcat.py` | 24GB VRAM Recommended |
| LongCat Image Edit | `python train_longcat_edit.py` | 24GB VRAM Recommended |

### New Architecture Backend Services
The new architecture uses a distributed service approach:

| Service | Command | Port | Purpose |
|---------|---------|------|---------|
| API Service | `python -m services.api_service.main` | 8000 | Handles HTTP requests and job queuing |
| Worker Service | `python -m services.worker_service.main` | N/A | Executes training jobs |
| Streamer Service | `python -m services.streamer_service.main` | 8001 | Streams real-time output to WebSocket clients |
| Combined Services | `python main_services.py` | 8000, 8001 | Runs all services together |

### Node-Based Frontend UI (Recommended)
For the new Node-Based Frontend UI with visualization capabilities:

**Development Mode (Fastest for development):**
```bash
# Terminal 1: Start new architecture backend services
python main_services.py

# Terminal 2: Start frontend (auto-reloads on changes)
cd frontend
npm run dev
```
Access at: http://localhost:3000

**Production Mode (Optimized for performance):**
```bash
# Build and serve the frontend with backend
python main.py
```
Access at: http://localhost:7860

**Preview Mode (Pre-built optimized version):**
```bash
# Terminal 1: Start new architecture backend services
python main_services.py

# Terminal 2: Serve pre-built frontend (faster than main.py)
cd frontend
npm run preview
```
Access at: http://localhost:7860

> **Performance Note**: `npm run dev` provides the fastest experience with hot reloading, while `npm run preview` offers optimized performance similar to production. The `python main.py` approach uses `npm run preview` internally for better performance but still requires the backend to be running separately.


## 🔧 Parameter Configuration Guide
[![CivitAI Article](https://img.shields.io/badge/📖-Detailed_Parameter_Guide-purple)](https://civitai.com/articles/7743)

---
## 🌌 Qwen Model Management
| Config | Usage |
|--------|-------|
| `config_qwen_single.json` | Train qwen image with a single image; leave the suffix empty to use all images without a suffix. |

- Usage: `python train_qwen_image.py --config_path config_qwen_single.json`


| Config | Usage |
|--------|-------|
| `config_qwen_single.json` | Train Qwen Image/Edit with a single image; leave the suffix empty to use all images without a suffix. |
| `config_qwen_edit_pairs.json` | Traditional Qwen Edit training using `_T` and `_R` suffixed images. |
| `config_qwen_edit_pairs_multiple.json` | Train with multiple reference images by setting suffixes like `_T`, `_R`, and `_G`. |

- Usage: `python train_qwen_image_edit.py --config_path config_qwen_single.json`

### Qwen Model Installation
Inpainting Model Setup
```bash
  hf download"lrzjason/qwen_image_nf4" --local-dir qwen_models/qwen_image_nf4/
```
For more details (example dataset):
- https://github.com/lrzjason/T2ITrainer/blob/main/doc/qwen.md

## ⚙️ Qwen Recommended Parameters
## Qwen Image NF4
| Category          | Settings                      |
|-------------------|-------------------------------|
| Base Configuration| Rank 32, AdamW, Learn Rate 1e-4       |
| 24GB GPU          | 512 resolution, Batch Size 1  |
| Precision         | bf16      |

## Qwen Image Model
| Category          | Settings                      |
|-------------------|-------------------------------|
| Base Configuration| Rank 32~64, AdamW, Learn Rate 1e-4       |
| 48GB GPU          | 1024 resolution, Batch Size 1  |
| Precision         | bf16      |

## Qwen Edit Model
| Category          | Settings                      |
|-------------------|-------------------------------|
| Base Configuration| Rank 32~64, AdamW, Learn Rate 1e-4       |
| 48GB GPU          | 512 resolution, Batch Size 1  |
| Precision         | bf16      |

## 💻 VRAM Usage (nf4, bs1,  blocks_to_swap=20)
<div align="center">
  <table>
    <tr>
      <td align="center">
        <strong>VRAM Peak</strong><br>
        <img src="https://github.com/lrzjason/T2ITrainer/blob/main/doc/image/qwen_nf4_block_swap_20.png" width="400">
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
        <img src="https://github.com/lrzjason/T2ITrainer/blob/main/doc/image/qwen_nf4_block_swap_0.png" width="400">
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

| Config | Usage |
|--------|-------|
| `config_new_single.json` | Train Kontext with a single image; leave the suffix empty to use all images without a suffix. |
| `config_new_pairs.json` | Traditional Kontext training using `_T` and `_R` suffixed images. |
| `config_new_pairs_multiple.json` | Train with multiple reference images by setting suffixes like `_T`, `_R`, and `_G`. |
| `config_new_mixed.json` | Train Kontext using a mixed layout—e.g., combine traditional pair training with single-image training. |

- Usage: `python train_flux_lora_ui_kontext_new.py --config_path config_new_single.json`

### Kontext Model Installation
Inpainting Model Setup
```bash
  hf download"lrzjason/flux-kontext-nf4" --local-dir flux_models/kontext/
```
For more details (example dataset):
- https://github.com/lrzjason/T2ITrainer/blob/main/doc/image/flux_kontext.md
- https://huggingface.co/datasets/lrzjason/object_removal_alpha_kontext

### Fill Model Installation (Skip if train kontext)
Inpainting Model Setup
```bash
  hf download"lrzjason/flux-fill-nf4" --local-dir flux_models/fill/ 
```
For more details (example dataset):
- https://github.com/lrzjason/T2ITrainer/blob/main/doc/image/flux_fill.md
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

## 🌌 LongCat Model Management

| Config | Usage |
|--------|-------|
| `config_longcat_dev.json` | Train LongCat Image with a single image; leave the suffix empty to use all images without a suffix. |
| `config_longcat_edit.json` | Train LongCat Image Edit with paired images using various suffixes like `_T`, `_R`, etc. |

- Usage (LongCat Image): `python train_longcat.py --config_path config_longcat_dev.json`
- Usage (LongCat Image Edit): `python train_longcat_edit.py --config_path config_longcat_edit.json`

### LongCat Model Installation
LongCat Model Setup
```bash
  hf download "Meituan/LongCat-Image" --local-dir longcat_models/LongCat-Image/
  hf download "Meituan/LongCat-Image-Edit" --local-dir longcat_models/LongCat-Image-Edit/
```

## ⚙️ LongCat Training Recommended Parameters
| Category          | Settings                      |
|-------------------|-------------------------------|
| Base Configuration| Rank 32~64, AdamW, Learn Rate 1e-4       |
| 24GB GPU          | 1024 resolution, Batch Size 1  |
| Precision         | bf16      |

## 💻 VRAM Usage nf4
<div align="center">
  <table>
    <tr>
      <td align="center">
        <strong>VRAM Peak</strong><br>
        <img src="https://github.com/lrzjason/T2ITrainer/blob/main/doc/image/nf4_example.png" width="400">
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
        <img src="https://github.com/lrzjason/T2ITrainer/blob/main/doc/image/fill_example_peak.png" width="400">
      </td>
      <td align="center">
        <strong>VRAM Low</strong><br>
        <img src="https://github.com/lrzjason/T2ITrainer/blob/main/doc/image/fill_example_low.png" width="400">
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

- **Configuration Guide**: [📖 CivitAI Article](https://civitai.com/articles/7743)


## 🆘 Troubleshooting
- **Kolors Black Image Issue**: Ensure you're using [FP16 Fixed VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)
- **VRAM Limitations**: Adjust `blocks_to_swap` parameter (higher values reduce memory usage)
- **Windows DLL Errors**: Verify VC++ Redistributable installation
- **Frontend Not Loading**: Ensure Node.js is installed and frontend is built (`cd frontend && npm install && npm run build`)
- **Templates Not Found**: In production builds, ensure the backend is running (`python backend_api.py`) before accessing the frontend
- **Slow Frontend Performance**: Use `npm run dev` for development or `npm run preview` for optimized local serving instead of `python main.py`
---


## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=lrzjason/T2ITrainer&type=Date)](https://star-history.com/#lrzjason/T2ITrainer&Date)

## Old Change logs: 
- https://github.com/lrzjason/T2ITrainer/blob/main/doc/change_logs.md

## Recent Change Logs:
- **2025-07-30:** 
- **Fix**: Remove text attention mask in lora training.

## Sponsor:
* Thanks to all the contributors and sponsors for improving the project!
* Sponsor List:
* https://github.com/lrzjason/T2ITrainer/blob/main/sponsor/sponsor_list.txt

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
- Thanks to 猫不爱吃香菜 sponsor for adding lokr support.
- Thanks to AIGate(https://waas.aigate.cc/) providing compute power for the development.
  


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
