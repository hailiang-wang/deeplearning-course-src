'''
数学公式推导：
note_5_2.md

'''
import numpy as np
import torch

torch.set_printoptions(edgeitems=2, linewidth=75)

###########################
# 给出未知单位的温度计的温度，让模型告诉我们，对应的摄氏温度是多少？
###########################

# X: 采集到的未知单位的温度
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]

# Y: 对应的摄氏度，目标计算的值
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]

t_u = torch.tensor(t_u) * 0.1
t_c = torch.tensor(t_c)


def model(x, w, b):
    '''
    模型，有两个参数，线性回归
    '''
    return (w * x) + b


def loss_fn(t_p, t_c):  # p --> predict
    '''
    损失函数，计算理想输出和实际输出的差异: 返回一个数值
    t_p 和 t_c 都是向量，代表多个样本
    '''
    squared_diffs = (t_p - t_c)**2  # 所有样本的损失的向量
    return squared_diffs.mean()


w = torch.ones(())  # tensor: scalar(数值, 1.0)， vector ([1,2]), matrix([[1,2], [3,4]]), ....
b = torch.zeros(())

t_p = model(t_u, w, b)

print(t_p)
# tensor([35.7000, 55.9000, 58.2000, 81.9000, 56.3000, 48.9000, 33.9000,
#        21.8000, 48.4000, 60.4000, 68.4000])

loss = loss_fn(t_p, t_c)
print("Loss,", loss)

###############################
# 思考：如何更新 w 和 b, 然后，保证下一次的运算，loss 可以更小？
# 两个核心的算法：梯度下降，反向传播
###############################

# 1.手动演算的过程
# delta = 0.1

# loss_rate_change_w = (loss_fn(model(t_u, w + delta, b), t_c) - loss_fn(model(t_u, w - delta, b), t_c)) / (2 * delta)

# loss_rate_change_b = (loss_fn(model(t_u, w, b + delta), t_c) - loss_fn(model(t_u, w, b - delta), t_c)) / (2 * delta)


# print(loss_rate_change_w)

# learning_rate = 1e-2

# w = w - loss_rate_change_w * learning_rate
# b = b - loss_rate_change_b * learning_rate


# 2. 让变化率无限小，无限逼近 0，那么就是导数
# 含有多个参数，每个参数的导数值，构成的向量，就是梯度
# 沿着梯度下降的方向，可以让 loss 函数降低：是因为，我们选择的 loss 损失函数，有凸函数的性质，即连续可导的凸函数，在导数等于 0 的时候，到达极值点

def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c) / t_p.size(0)
    return dsq_diffs

# dloss_fn 是关于loss 函数和输入 t_p 之间的导数，而我们需要的是 loss 和 (w,b) 之间的导数，
# 而 t_p 和 w,b 之间是通过 model(w*x + b) 实现的，进而，通过链式法则：关于 dloss_fn, 和 model(w*x + b) 就可以求出 loss 和 (w,b) 之间的导数


def dmodel_dw(t_u, w, b):
    return t_u


def dmodel_db(t_u, w, b):
    return 1.0


def grad_fn(t_u, t_c, t_p, w, b):
    '''
    求解梯度
    1）梯度包括两个参数组成的导数的向量：loss 函数是关于 w, b 的函数
    2）w,b 也是 model 的参数；
    3）基于以上，使用微积分的链式法则进行求导：

        loss --> model --> w
        loss --> model --> b

        我们为了找到 loss 关于 w,b 的导数，需要以 model 为桥梁

    导数基本知识：
    1）公式大全 - https://zhuanlan.zhihu.com/p/464253616
    2）链式法则 - https://zhuanlan.zhihu.com/p/665621495

    '''
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w, b)

    return torch.stack([dloss_dw.sum(), dloss_db.sum()])


# 开始训练，使用自动学习参数的机制：应用梯度求解变化率，多轮的更新
learning_rate = 1e-3
epochs = 100000

for epoch in range(1, epochs + 1):
    t_p = model(t_u, w, b)
    loss = loss_fn(t_p=t_p, t_c=t_c)

    grad = grad_fn(t_u, t_c, t_p, w, b)

    w = w - (learning_rate * grad[0])
    b = b - (learning_rate * grad[1])

    print('Epoch %d, Loss %f' % (epoch, float(loss)))  # <3>

    if not torch.isfinite(loss).all():
        break

print("Result:")
print(w)
print(b)

from matplotlib import pyplot as plt

t_p = model(t_u, w, b)  # <1>

fig = plt.figure(dpi=600)
plt.xlabel("Temperature (°Fahrenheit)")
plt.ylabel("Temperature (°Celsius)")
plt.plot(t_u.numpy(), t_p.detach().numpy())  # <2>
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.savefig("temp_unknown_plot.png", format="png")

plt.show()
