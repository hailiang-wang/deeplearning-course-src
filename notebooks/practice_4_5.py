import numpy as np
import torch

torch.set_printoptions(edgeitems=3, threshold=30)

lines = []
with open("data/p1ch4/jane-austen/1342-0.txt", "r", encoding="utf-8") as fin:
    lines = fin.readlines()[:10]

contents = []
for l in lines:
    l = l.strip()

    if l:
        print(l)
        contents.append(l)

print(len(contents))

l = contents[0]
letter_t = torch.zeros(len(l), 128)

for i, letter in enumerate(l.lower().strip()):
    letter_index = ord(letter) if ord(letter) < 128 else 0
    letter_t[i][letter_index] = 1

print(letter_t)

'''
问题：对于将文本处理为张量？onehot 有缺点：
* 稀疏矩阵
* 只有 0，1，非常的消耗空间
* 不能通过 cosin 距离，得到两个词之间的语义相似度
'''

# 什么叫做 cosin 距离？
# 不同的两个向量，在空间中的夹角 \theta 的 cosin 值代表相似度
# 特点：如果一个向量，是可以通过伸缩成为另一个向量，那么两个相似度就是 1；如果两个向量是垂直的，那么相似度就是 0


import torch
from torch import nn
cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

input1 = torch.tensor([1, 0]).float()  # 苹果
input2 = torch.tensor([0.5, 1]).float()  # 橘子

# ~ 0.5
output = cos(input1, input2)
print(output)
