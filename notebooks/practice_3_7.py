'''
张量的持久化
'''

import torch
# torch.manual_seed(100)

# x = torch.randn(3, 4, device=torch.device("cpu"))
# torch.save(x, "x.pth")
# print(x)

x = torch.load("x_cpu.pth")
x = torch.load("x_gpu.pth")  # 如果机器上没有 GPU，那么这行代码，能成功吗？
