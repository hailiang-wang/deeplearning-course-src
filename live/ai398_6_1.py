'''
使用 nn linear 和 optimimzer
'''
import torch
import torch.optim as optimimzer
import torch.nn as nn
import sys
import random
import numpy as np

torch.set_printoptions(edgeitems=2, linewidth=75)

torch.manual_seed(100)
random.seed(100)
np.random.seed(100)

###########################
# 给出位置单位的温度计的温度，让模型告诉我们，对应的摄氏温度是多少？
###########################

# X: 采集到的位置单位的温度
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]

# Y: 对应的摄氏度，目标计算的值
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]

t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u) * 0.1

##################
# 分数据集为两个部分：
# 1) 训练集
# 2）验证集
##################
n_samples = t_c.shape[0]
n_validate = int(0.1 * n_samples)

shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_validate]
validate_indices = shuffled_indices[-n_validate:]

print("train", train_indices)
print("valid", validate_indices)

train_data_x = t_u[train_indices].unsqueeze(1)
train_data_y = t_c[train_indices]

validate_data_x = t_u[validate_indices].unsqueeze(1)
validate_data_y = t_c[validate_indices]
print("validate_data_x", validate_data_x)
print("validate_data_y", validate_data_y)

'''
搭建模型
'''
model = torch.nn.Linear(in_features=1, out_features=1, bias=True)

for name, params in model.named_parameters():
    print(name, params)


def loss_fn(t_p: torch.Tensor, t_c: torch.Tensor) -> torch.Tensor:  # p --> predict
    '''
    损失函数，计算理想输出和实际输出的差异: 返回一个数值
    t_p 和 t_c 都是向量，代表多个样本
    '''
    t_p = t_p.squeeze(dim=1)
    # print("loss fn tp shape", t_p.shape)
    # print("loss fn tc shape", t_c.shape)

    squared_diffs = (t_p - t_c)**2  # 所有样本的损失的向量
    return squared_diffs.mean()


# 开始训练，使用自动学习参数的机制：应用梯度求解变化率，多轮的更新
learning_rate = 1e-2
opt = optimimzer.SGD(model.parameters(), lr=learning_rate)
epochs = 10000

for epoch in range(1, epochs + 1):
    t_p = model(train_data_x)
    loss = loss_fn(t_p=t_p, t_c=train_data_y)

    opt.zero_grad()
    loss.backward()

    # print("  w loss(%s), b loss(%s)" % (w.grad.item(), b.grad.item()))
    with torch.no_grad():
        opt.step()

    print('Epoch %d, Train Loss %f' % (epoch, float(loss)))

    if not torch.isfinite(loss).all():
        break

    if epoch % 100 == 0:
        with torch.no_grad():

            # print("model(valid_data_x)", model(valid_data_x).shape)
            # print("valid_data_y", valid_data_y.shape)

            valid_loss = loss_fn(model(validate_data_x), validate_data_y)
            print('Epoch %d, Validate Loss %f' % (epoch, float(valid_loss)))

print("Result:")

for param in model.parameters():
    print(param.item())

from matplotlib import pyplot as plt

t_p = model(t_u.unsqueeze(1))
print("t_p shape", t_p.shape)

fig = plt.figure(dpi=600)
plt.xlabel("Temperature (°Fahrenheit)")
plt.ylabel("Temperature (°Celsius)")
plt.plot(t_u.numpy(), t_p.detach().numpy())  # <2>
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.savefig("temp_unknown_plot.png", format="png")

plt.show()
