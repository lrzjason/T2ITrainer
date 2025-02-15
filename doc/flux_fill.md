WIP

Object Removal Alpha
Example dataset:
https://huggingface.co/datasets/lrzjason/ObjectRemovalAlpha

For training dataset of flux fill lora, 
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
