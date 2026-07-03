'''
绘图参考：assets/梯度下降_理解.pdf
'''

import torch

torch.set_printoptions(edgeitems=2, threshold=30)

###########################
# 给出未知单位的温度计的温度，让模型告诉我们，对应的摄氏温度是多少？
###########################

# X: 采集到的未知单位的刻度
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]

# Y: 对应的摄氏度，目标计算的值
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]

t_u = torch.tensor(t_u, dtype=torch.float32) * 0.1
t_c = torch.tensor(t_c, dtype=torch.float32)


def model(x, w, b):
    return (w * x) + b


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()


'''
人为的选择两个直线，进行比较
'''


# # 直线2
# w2 = 10.0
# b2 = 20.0
# t_p2 = model(t_u, w2, b2)
# print("Loss 2", loss_fn(t_p2, t_c))

# 随机初始化的 w, b
w1 = 1.0
b1 = 10.0
t_p1 = model(t_u, w1, b1)
loss1 = loss_fn(t_p1, t_c)
print("Loss 1", loss1)

# 当我们得到损失以后, 下一步，该怎么改变w， b，以使得 loss 变小呢?
# 比如，在分析这个问题的时候，我们先固定 b，让 w 变化
# 得到了一个曲线

delta_w = 1e-5
learning_rate = 1e-3

w2 = w1 + delta_w
w3 = w1 - delta_w

# 我们想了解，在 delta w 的变化上，对 loss 的影响：也就是 loss 的辩护率，是什么样的？

loss_w2 = loss_fn(model(t_u, w2, b1), t_c)
loss_w3 = loss_fn(model(t_u, w3, b1), t_c)

# change_rate = (loss_w2 - loss_w3) / (w2 - w3)
change_rate = (loss_w2 - loss_w3) / (2 * delta_w)

# 启发1：如果 change rate（loss 函数的变化率）是大于 0 的，那么，w 就向左侧更新 delta_w，就可以保证更新后的 w 和 b 的损失loss 是降低的！
if change_rate > 0:
    w1 = w1 - delta_w
elif change_rate < 0:
    # 启发2：如果 change rate 小于 0，那么 w 应该向右侧移动 delta_w
    w1 = w1 + delta_w
else:
    # change_rate = 0
    pass

# change_rate ?
# 当 delta --> 0，也就是 delta_w 无限的接近 0，那么 change_rate 就是导数!
# 不管是启发1还是启发2，那么 w1 更新的方向，都是可以用下面表达式表达：
w1 = w1 - (change_rate * learning_rate)
# b1 = b1 - (change_rate_b)  # 使用上述 delta，在求解一个 change_rate_b

# 重复上面的过程，直到 loss 不再变化，学习就停止了
