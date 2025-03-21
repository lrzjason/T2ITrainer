import torch

device = torch.cuda.get_device_capability(0)
print("device:")
print(device)