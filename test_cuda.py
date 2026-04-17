import torch

print('cuda', torch.version.cuda)
print('available', torch.cuda.is_available())