

import torch

img_t = torch.randn((3, 5, 5))  # 单张图片
batch_t = torch.randn((2, 3, 5, 5))  # 多张图片/batch, channels, width, hights

weights = torch.tensor([0.2126, 0.7152, 0.0722])  # vector
print("weights.shape", weights.shape)
# 我们想让 5x5 的矩阵，分别有 3 个：R 红色，G 绿色，B 蓝色

img_gray_naive = img_t.mean(-3)
# 3x5x5 -mean(-3)-> img_gray_naive.shape？

batch_t_gray_naive = batch_t.mean(-3)
# 2x3x5x5 -mean(-3)-> batch_t_gray_naive.shape？ 2x5x5

weights_unsqueezed = weights.unsqueeze(-1).unsqueeze(-1)

print("weights_unsqueezed.shape", weights_unsqueezed.shape)

img_weights = img_t * weights_unsqueezed  # 3x[5x5] x 3x[1x1]
print("img_t", img_t.shape)
print("img_t\n", img_t)
print("\n\nimg_weights", img_weights.shape)
print("img_weights\n", img_weights)
