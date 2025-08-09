# 🎨 Qwen Training Guide
**Requires diffusers>=0.35.0**

## Download Qwen Image diffusers structure models
For 24GB GPU
```bash
hf download"lrzjason/qwen_image_nf4" --local-dir qwen_models/qwen_image_nf4/
```

For 48GB GPU
```bash
hf download"Qwen/Qwen-Image" --local-dir qwen_models/qwen_image_nf4/
```


## 📂 Training Dataset Requirements
Image pairs must include:
- `*.png | *.webp | *.jpg` : Training image
- `*.txt` : Text prompt, the name should be the same as the image name

Folder structure example:
```example
    root/
    ├── man/
    │   ├── 1.jpg
    │   ├── 1.txt
    │   ...
    └── woman/
        ├── 1.jpg
        ├── 1.txt
        ...
```
or
```example
    root/
    ├── 1.jpg
    ├── 1.txt
    ...
```


## ⚙️ Recommended Parameters
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

## 🚀 Launch Options
| Model          | Command                  | Special Notes                     |
|-----------------|--------------------------|-----------------------------------|
| Qwen Image    | `python train_qwen_image.py` | Requires diffusers>=0.35.0, 24GB VRAM Recommended for nf4, 48GB VRAM Recommended for original |

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
