import torch


if torch.cuda.is_available():
    print("cuda")
else:
    print("cpu")