'''
P169/446 6.5 练习题

加载第4章中的葡萄酒数据集，并使用适当数量的输人参数创建一个新模型。
a）与我们使用的温度数据相比，需要多长时间来训练？
b）你能解释一下影响训练时间的因素吗？
c）在对这个数据集进行训练时，你能减小损失吗？
d）你将如何绘制此数据集的图形？
'''
import sys
import numpy as np
import torch
import csv

torch.set_printoptions(edgeitems=3, threshold=30)

wine_path = "data/p1ch4/tabular-wine/winequality-white.csv"
wine_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)

col_list = next(csv.reader(open(wine_path), delimiter=";"))
print(wine_numpy.shape, col_list)

'''
加载原始数据
'''
wineq = torch.from_numpy(wine_numpy)
print("wineq shape", wineq.shape)
print("wineq dtype", wineq.dtype)

data = wineq[:, :-1]
target = wineq[:, -1].long()

print("data shape %s, target %s" % (data.shape, target.shape))
# 10 分类问题，0~9：0 是最低分，9 是最高分

'''
处理 Y 理想输出为独热编码 One Hot Encoding
在分类问题中，经常的使用
'''
target_onehot = torch.zeros(target.shape[0], 10)
# print("target shape", target.shape)  # [4898]
# print("target_onehot shape", target_onehot.shape)  # 4898x10

# Y
target_onehot_index = target.unsqueeze(1)
print("target_onehot_index", target_onehot_index)
print("target_onehot_index shape", target_onehot_index.shape)
target_onehot.scatter_(1, target_onehot_index, 1.0)


# 将数据集拆分成两个部分：train_data 和 valid_data
torch.manual_seed(100)
n_samples = data.shape[0]
n_val = int(n_samples * 0.125)
shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
valid_indices = shuffled_indices[-n_val:]
train_data_x = data[train_indices]
train_data_y = target_onehot[train_indices]

valid_data_x = data[valid_indices]
valid_data_y = target[valid_indices]

print("train_data_x shape", train_data_x.shape)
print("train_data_y shape", train_data_y.shape)


'''
定义模型
'''
model = torch.nn.Sequential(
    torch.nn.Linear(11, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 10),
)

epoches = 10000
lr = 1e-3
opt = torch.optim.SGD(model.parameters(), lr=lr)
loss_fn = torch.nn.MSELoss()

for i in range(epoches):
    opt.zero_grad()
    y_p = model(train_data_x)
    # print(y_p.shape)
    loss = loss_fn(y_p, train_data_y)
    loss.backward()
    opt.step()

    print("Epoch %s train/loss %.4f" % (i, loss))

'''
在验证集上预测
'''

target_predict = model(valid_data_x)
# 转化为分类
q = torch.argmax(target_predict, dim=-1)
print(q)
matched = (q.long() == valid_data_y).sum()
print("准确预测：", matched.item(), "预测的种类总数", q.shape[0])
print("准确率 %.2f%%" % ((matched.item() / q.shape[0]) * 100))
