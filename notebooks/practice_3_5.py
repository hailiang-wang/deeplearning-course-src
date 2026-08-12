'''
命名张量
pip install pillow
'''
import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))

import torch
torch.manual_seed(100)
from PIL import Image  # Image 可以建立成单通道的吗？
import numpy as np

img_path = os.path.join(curdir, "data", "p1ch2", "bobby.jpg")
img = Image.open(img_path)
img_array = np.asarray(img)
print(img_array.shape)  # (720, 1280, 3) (高，寬，RGB 通道) --> 轉換爲 Tensor 張量
img_t = torch.tensor(img_array.tolist()).permute(2, 1, 0)  # 3x1280x720
print(img_t.shape)

'''
题目：
将 img_t 转换为灰度的，并且使用权重
weights = torch.tensor([0.2126, 0.7152, 0.0722])

最终，再存储回一张图片，路径 img_path2
'''
img_path2 = os.path.join(curdir, "bobby_gray.jpg")

# 你的代码


# img_t = torch.randn(3, 5, 5)  # shape [channels, rows, columns]
# weights = torch.tensor([0.2126, 0.7152, 0.0722])

# batch_t = torch.randn(2, 3, 5, 5)

# '''
# 计算统计数据
# '''
# print(img_t)
# img_gray_naive = img_t.mean(dim=-3)  # 1x5x5
# print(img_gray_naive)
# print(img_gray_naive.shape)

# batch_gray_navie = batch_t.mean(dim=-3)

# '''
# 根据指定的权重，进行计算
# '''
# unsqueezed_weights = weights.unsqueeze(-1).unsqueeze(-1)
# # print(unsqueezed_weights.shape)  # torch.Size([3, 1, 1])
# # img_t 3x5x5 -> 1x5x5
# img_gray_weighted = (img_t * unsqueezed_weights).sum(dim=-3)

# # batch_t 2x3x5x5, unsqueezed_weights 3x1x1
# batch_gray_weighted = (batch_t * unsqueezed_weights).sum(dim=-3)
