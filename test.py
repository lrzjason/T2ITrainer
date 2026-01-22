import torch

# 方法1：直接打印版本
print(torch.__version__)

# 方法2：详细版本信息
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
print(f"cuDNN 版本: {torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A'}")