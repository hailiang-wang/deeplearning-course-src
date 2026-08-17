import numpy as np
import torch
from torch import Tensor

torch.set_printoptions(edgeitems=3, threshold=30)

###########################
# 给出未知单位的温度计的温度，让模型告诉我们，对应的摄氏温度是多少？
###########################

# X: 采集到的未知单位的温度
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]

# Y: 对应的摄氏度，目标计算的值
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]

t_u = torch.tensor(t_u) * 0.1
t_c = torch.tensor(t_c)

'''
定义模型
'''


def model(t_u: Tensor, w: Tensor, b: Tensor) -> Tensor:
    '''
    一元一次线性回归模型
    '''
    return w * t_u + b


def loss_fn(t_p: Tensor, t_c: Tensor) -> Tensor:
    '''
    MSE 均方误差损失函数
    '''
    squared_diff = (t_p - t_c)**2
    return squared_diff.mean()


'''
自动求导的使用
https://zhuanlan.zhihu.com/p/2027423168176899509
'''
params = torch.tensor([0.5, 0.0], requires_grad=True)

'''
执行训练
'''
epoches = 20000
learning_rate = 1e-3

for epoch in range(1, epoches + 1):
    t_p = model(t_u, params[0], params[1])

    if params.grad is not None:
        params.grad.zero_()

    loss = loss_fn(t_p, t_c)
    loss.backward()

    with torch.no_grad():
        # params[0] -= (learning_rate * params.grad[0])
        # params[1] -= (learning_rate * params.grad[1])
        params -= (learning_rate * params.grad)

    print("Epoch %s, Loss %.4f" % (epoch, loss.item()))
    if not torch.isfinite(loss).all():
        break

print("Result:")
print(params)


from matplotlib import pyplot as plt

t_p = model(t_u, params[0], params[1])  # <1>

fig = plt.figure(dpi=600)
plt.xlabel("Temperature (°Fahrenheit)")
plt.ylabel("Temperature (°Celsius)")
plt.plot(t_u.numpy(), t_p.detach().numpy())  # <2>
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.savefig("temp_unknown_plot.png", format="png")

plt.show()
