##Flux Fill Training

##Object Removal Alpha Example  
Example dataset:  
https://huggingface.co/datasets/lrzjason/ObjectRemovalAlpha  

Trained Lora:  
https://civitai.com/models/1121321/object-removal-flux-fill

##For training dataset of flux fill lora,  
it requires image pairs including  
- Ground True image suffix with '_G'
- Factual image suffix with '_F'
- Mask image suffix with '_M'

Each pairs should be in the same folder.  
But you could have multiple subfolders in one training.  
For example:  
```
root/
- man/
    - 1_G.jpg
    - 1_F.jpg
    - 1_M.jpg
    ...
- woman/
    - 1_G.jpg
    - 1_F.jpg
    - 1_M.jpg
    ...
```

##Reference Parameters:
For flux fill training, it is recommended to use the following parameters:
- rank 32
- adamw
- lr 1e-4

For 24 GB GPU
- resolution 512 
- batch size 1
- blocks_to_swap 10
- for lower GPU, please use more blocks_to_swap like: 15 or 20
- mask_dropout = ignore mask area, all pixels viewed as masked when x ratio.
- mixed precision training, it is recommended to use bf16 on 3090. You could use fp8 on 40xx device.


## Contact
- **Twitter**: [@Lrzjason](https://twitter.com/Lrzjason)  
- **Email**: lrzjason@gmail.com  
- **QQ Group**: 866612947  
- **Civitai**: [xiaozhijason](https://civitai.com/user/xiaozhijason)


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
