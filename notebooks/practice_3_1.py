'''
使用 Pytorch 创建张量
'''
import torch

# 创建一个行向量
x: torch.Tensor = torch.arange(12)
# x = torch.arange(4, 12)
print(x)
print(x.shape)
print(x.numel())

# 使用 reshape 创建一个多维度的 tensor
y = x.reshape((2, 6))
print(y.shape)
print(y)

z = x.reshape((2, 2, -1))
print(z.shape)
print(z)
