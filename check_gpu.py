import torch
print("Is CUDA Available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())