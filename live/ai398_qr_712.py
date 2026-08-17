import numpy as np
import torch
import torch.optim as optimimzer
import torch.nn as nn
import sys
import random
import numpy as np

torch.set_printoptions(edgeitems=2, precision=2, linewidth=75)

torch.manual_seed(100)
random.seed(100)
np.random.seed(100)

import csv
wine_path = "data/p1ch4/tabular-wine/winequality-white.csv"
wine_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)

wineq = torch.from_numpy(wine_numpy)

data = wineq[:, :-1]      # 所有行，前11列 → 特征
target = wineq[:, -1].long()  # 最后一列 → 标签（质量评分），转为长整型

target_onehot = torch.zeros(target.shape[0], 10)   # 预设 10 个类别（0~9）
tu = target.unsqueeze(1)                          # 变为 (N, 1) 用于 scatter_
target_onehot.scatter_(1, tu, 1.0)

data_mean = torch.mean(data, dim=0)   # 按列求均值，形状 (11,)
data_std = torch.std(data, dim=0)    # 按列求标准差
data_norm = (data - data_mean) / data_std

n_samples = data_norm.shape[0]               # 样本总数 = 11
n_validate = int(0.2 * n_samples)      # 验证集大小 = 1（因为 0.1*11 = 1.1，取整后为 1）

shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_validate]   # 前 80% 个作为训练
validate_indices = shuffled_indices[-n_validate:]  # 后 20% 个作为验证

train_data_x = data_norm[train_indices]
train_data_y = target_onehot[train_indices]

validate_data_x = data_norm[validate_indices]
validate_data_y = target_onehot[validate_indices]

model = torch.nn.Linear(in_features=11, out_features=10, bias=True)

# 损失函数
loss_fn = nn.MSELoss()

learning_rate = 1e-2
opt = optimimzer.SGD(model.parameters(), lr=learning_rate)   # SGD 优化器
epochs = 10000

for epoch in range(1, epochs + 1):
    t_p = model(train_data_x)                     # 前向传播，得到预测值
    # print("t_p shape", t_p.shape)
    # print("train_data_y shape", train_data_y.shape)

    loss = loss_fn(t_p, train_data_y)            # 计算训练损失

    opt.zero_grad()                              # 清空之前的梯度
    loss.backward()                              # 反向传播，计算梯度
    opt.step()                                   # 更新参数（执行一步 SGD）

    # 打印训练损失（每轮都打印）
    print('Epoch %d, Train Loss %f' % (epoch, float(loss)))

    # 检查损失是否出现 NaN 或 Inf，若是则提前停止
    if not torch.isfinite(loss).all():
        break

    # 每 100 轮计算一次验证损失（不追踪梯度）
    if epoch % 100 == 0:
        with torch.no_grad():
            valid_loss = loss_fn(model(validate_data_x), validate_data_y)
            print('Epoch %d, Validate Loss %f' % (epoch, float(valid_loss)))

print("训练完成后的模型参数：")
print("权重 (Weight):", model.weight.data)
print("偏置 (Bias):", model.bias.data)

######################################
# 练习： 计算上面模型在验证集的准确率
#      1) 准确率公式： 求和（识别的酒的分类，和真正这个酒所在的分类，是一样的）/ 预测的次数
######################################
