'''
介绍 NLL Loss 的使用
'''

import torch
import torch.nn as nn

log_softmax = nn.LogSoftmax(dim=1)
nll_loss = nn.NLLLoss()

predicts = torch.tensor([
    [2.0, 1.0, 0.1],  # 样本 1 的 Logits
    [0.5, 3.0, 1.5],  # 样本 2 的 Logits
    [1.0, 0.5, 2.5]   # 样本 3 的 Logits
])

targets = torch.tensor([0, 1, 2])  # 理想输出的标签的【向量】

'''
执行 loss 的生成
'''
log_probs = log_softmax(predicts)  # 第一步

print(log_probs)
print(log_probs.shape)

loss = nll_loss(log_probs, targets)  # 第二步
print("Loss", loss.item())

# loss.backward()
# opt.step()
