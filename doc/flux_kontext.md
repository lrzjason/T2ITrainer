# 🎨 Flux Kontext Training Guide
**Requires diffusers>=0.32.0**

### Kontext Model Installation
Kontext Model Setup
```bash
  huggingface-cli download "lrzjason/flux-kontext-nf4" --local-dir flux_models/kontext/
```
For more details (example dataset):
- https://github.com/lrzjason/T2ITrainer/blob/main/doc/flux_kontext.md
- https://huggingface.co/datasets/lrzjason/object_removal_alpha_kontext

### Folder Structure
<div align="center">
  <table>
    <tr>
      <td align="center">
        <p>flux_models:</p>
        <img src="https://github.com/lrzjason/T2ITrainer/blob/main/doc/image/folder_structure.png" alt="flux_models" width="400" />
      </td>
      <td align="center">
        <p>kontext:</p>
        <img src="https://github.com/lrzjason/T2ITrainer/blob/main/doc/image/kontext.png" alt="kontext" width="400" />
      </td>
    </tr>
  </table>
</div>

## 🔍 Object Removal Alpha Example
- **Example Dataset**:  
  [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-orange)](https://huggingface.co/datasets/lrzjason/object_removal_alpha_kontext)

## 📂 Training Dataset Requirements
Image pairs must include:
- `_T` suffix: Training image
- `_R` suffix: Reference

Folder structure example:
```example
    root/
    ├── dir1/
    │   ├── 1_T.jpg
    │   ├── 1_T.txt
    │   ├── 1_R.jpg
    └── dir2/
        ├── 1_T.jpg
        ├── 1_T.txt
        ├── 1_R.jpg
```

## 📂 Slider Training
Image pairs must include:
- `_T` suffix: Training image
- `_R` suffix: Reference

Enable use_two_captions (optional)
Folder structure example:
```example
    root/
    ├── dir1/
    │   ├── 1_T.jpg
    │   ├── 1_T.txt
    │   ├── 1_R.jpg
    │   ├── 1_R.txt
    └── dir2/
        ├── 1_T.jpg
        ├── 1_T.txt
        ├── 1_R.jpg
        ├── 1_R.txt
```

## ⚙️ Recommended Parameters
| Category          | Settings                      |
|-------------------|-------------------------------|
| Base Configuration| Rank 16, AdamW, Lr 1e-4       |
| 24GB GPU          | 512 resolution, Batch Size 1  |
| VRAM Optimization | nf4 repo                      | 
| Precision         | bf16 (3090)                   |

## 💻 VRAM Usage
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

## 🚀 Launch Options
| Script          | Command                  | Special Notes                     |
|-----------------|--------------------------|-----------------------------------|
| Flux kontext    | `python ui_flux_fill.py` | Requires diffusers>=0.32.0, 24GB VRAM Recommended |

## 📬 Contact
- 𝕏 **Twitter**: [@Lrzjason](https://twitter.com/Lrzjason)
- 📧 **Email**: [lrzjason@gmail.com](mailto:lrzjason@gmail.com)
- 💬 **QQ Group**: 866612947
- 🎨 **CivitAI**: [xiaozhijason](https://civitai.com/user/xiaozhijason)

## ☕ Support Development
<div align="center">
  <table>
    <tr>
      <td align="center">
        <strong>Buy Me Coffee</strong><br>
        <img src="https://github.com/lrzjason/Comfyui-In-Context-Lora-Utils/blob/main/image/bmc_qr.png" width="200">
      </td>
      <td align="center">
        <strong>WeChat</strong><br>
        <img src="https://github.com/lrzjason/Comfyui-In-Context-Lora-Utils/blob/main/image/wechat.jpg" width="200">
      </td>
    </tr>
  </table>
</div>
