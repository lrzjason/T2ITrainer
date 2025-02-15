# 🎨 Flux Fill Training Guide
**Requires diffusers>=0.32.0**

## 🔍 Object Removal Alpha Example
- **Example Dataset**:  
  [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-orange)](https://huggingface.co/datasets/lrzjason/ObjectRemovalAlpha)
- **Trained LoRA**:  
  [![CivitAI](https://img.shields.io/badge/🖼️-CivitAI-blue)](https://civitai.com/models/1121321/object-removal-flux-fill)

## 📂 Training Dataset Requirements
Image pairs must include:
- `_G` suffix: Ground Truth image
- `_F` suffix: Factual image  
- `_M` suffix: Mask image

Folder structure example:
```example
    root/
    ├── man/
    │   ├── 1_G.jpg
    │   ├── 1_F.jpg
    │   └── 1_M.jpg
    └── woman/
        ├── 1_G.jpg
        ├── 1_F.jpg
        └── 1_M.jpg
```

## ⚙️ Recommended Parameters
| Category          | Settings                      |
|-------------------|-------------------------------|
| Base Configuration| Rank 32, AdamW, LR 1e-4       |
| 24GB GPU          | 512 resolution, Batch Size 1  |
| VRAM Optimization | blocks_to_swap: 10            | (15-20 for lower VRAM GPUs) 
| Precision         | bf16 (3090) / fp8 (40xx)      |

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
