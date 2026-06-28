'''
自动求导
'''
import numpy as np
import torch
torch.set_printoptions(edgeitems=2, linewidth=75)

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u) * 0.1


def model(t_u, w, b):
    return w * t_u + b


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()  # 返回值是一个张量


params = torch.tensor([1.0, 0.0], requires_grad=True)
if params.grad is None:
    print("Params is None after init.")


# loss = loss_fn(model(t_u, *params), t_c)
# PyTorch 会发现 loss 的计算过程中，依赖了 params, 并且 params 的 requires_grad 设置上了: 那么，params 就会成为计算图中的一个节点，并非可以追踪回来，得到一个计算的数据公式
# 这个数据公式，是由调用 backward 的张量，方向可以计算相对应的导数的。
# loss.backward()  # 反向计算梯度

'''
自动求导
https://zhuanlan.zhihu.com/p/2027423168176899509
'''
# params.grad


'''
利用自动求导，实现模型的参数评估
'''

epoch = 10000
lr = 1e-2

for i in range(epoch):
    t_p = model(t_u, *params)

    if params.grad is not None:
        params.grad.zero_()
    loss = loss_fn(t_p, t_c)
    print("Epcho %s, loss %.5f" % (i, loss.item()))

    loss.backward()

    with torch.no_grad():
        params -= (lr) * params.grad

print("Final result w: %s, b: %s" % (params[0].item(), params[1].item()))

'''
Draw the line for fitting data
'''
from matplotlib import pyplot as plt
t_p = model(t_u, *params)  # <1>

fig = plt.figure(dpi=600)
plt.xlabel("Temperature (°Fahrenheit)")
plt.ylabel("Temperature (°Celsius)")
plt.plot(t_u.numpy(), t_p.detach().numpy())  # <2>
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.savefig("temp_unknown_plot.png", format="png")  # bookskip
