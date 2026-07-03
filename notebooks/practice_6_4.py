'''
使用 NN Loss
'''
import torch
import torch.optim as optimizer
import torch.nn as nn
import sys
'''
控制台执行：
pip install torchinfo
'''
from torchinfo import summary

torch.set_printoptions(edgeitems=2, threshold=30)


'''
机器学习的训练，分成两个阶段：
1）使用少量的训练数据，寻找超参数 --> 在这个阶段，我们希望比较不同的超参数的好坏，就要一样的随机数
2）使用找到的超参数，进行全量的训练数据的训练 --> 模型训练好，就可以上线了
'''
torch.manual_seed(100)


###########################
# 给出位置单位的温度计的温度，让模型告诉我们，对应的摄氏温度是多少？
###########################

# X: 采集到的位置单位的温度
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]

# Y: 对应的摄氏度，目标计算的值
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]

t_u = torch.tensor(t_u, dtype=torch.float32) * 0.1
t_c = torch.tensor(t_c, dtype=torch.float32)

'''
将数据集分成 train/validate 两个部分
train:validate = 9:1
'''
n_samples = t_u.shape[0]
n_val = int(n_samples * 0.1)

shuffled_indices = torch.randperm(n_samples)
# print(shuffled_indices)

train_indices = shuffled_indices[:-n_val]
valid_indices = shuffled_indices[-n_val:]

# print(len(train_indices))
# print(len(valid_indices))
train_data_x = t_u[train_indices].unsqueeze(1)
train_data_y = t_c[train_indices]

# print("train x", train_data_x.tolist())
# print("train y", train_data_y.tolist())

valid_data_x = t_u[valid_indices].unsqueeze(1)
valid_data_y = t_c[valid_indices]

'''
构建训练的模型
'''

# 使用 nn module 构建了一个全连接的一个隐含层的神经网络
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
    ('hidden_linear', nn.Linear(1, 13)),
    ('hidden_activation', nn.ReLU()),
    ('output_layer', nn.Linear(13, 1))
]))

print(model)

'''
使用 named parameters 计算参数总量
'''
# total_params = 0
# for name, param in model.named_parameters():
#     print(name, param.shape)
#     total_params += param.numel()

# print("Total params ", total_params)

'''
使用 named parameters 去查看 参数
'''
print("weight params of hidden layer", model.hidden_linear.weight)
print("bias params of hidden layer", model.hidden_linear.bias)


# def loss_fn(t_p: torch.Tensor, t_c: torch.Tensor):
#     t_p = t_p.squeeze(dim=1)
#     return ((t_p - t_c)**2).mean()
'''
在 PyTorch 中，包含的常用的损失函数：
https://zhuanlan.zhihu.com/p/666303929
'''
loss_fn = nn.MSELoss()


# Hyper parameters
LR = 1e-3
EPOCH = 100000

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
opt = optimizer.SGD(model.parameters(), lr=LR)
torch.set_default_device(DEVICE)

for epoch in range(1, EPOCH + 1):

    t_p = model(train_data_x)
    loss = loss_fn(t_p, train_data_y.unsqueeze(dim=1))

    opt.zero_grad()
    loss.backward()
    opt.step()

    print("Epoch %s train/loss %.4f" % (epoch, loss.item()))

    if not torch.isfinite(loss).all():
        break

    if epoch % 100 == 0:
        with torch.no_grad():
            valid_loss = loss_fn(model(valid_data_x), valid_data_y.unsqueeze(dim=1))
            print("Epoch %s valid/loss %.4f" % (epoch, valid_loss.item()))

print("*" * 100)
for param in model.parameters():
    print(param)

from matplotlib import pyplot as plt

t_p = model(t_u.unsqueeze(dim=1))

fig = plt.figure(dpi=600)
plt.xlabel("Temperature (°Fahrenheit)")
plt.ylabel("Temperature (°Celsius)")
plt.plot(t_u.numpy(), t_p.detach().numpy())  # <2>
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.savefig("temp_unknown_plot.png", format="png")

plt.show()
