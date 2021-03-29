import torch.nn.functional as F

import torch
from torch import nn

x = torch.randn(1, 256, 64, 64).expand(1, 256, 64, 64)
print(x.shape)

k = torch.randn(6, 256, 1, 1)
y = F.conv2d(x, k, groups=6)
print(y.shape)
