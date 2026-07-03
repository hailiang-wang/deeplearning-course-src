'''
P169/446 6.5 练习题

在 practice_6_5_3.py 的基础上，使用 Mini Batch
'''
import sys
import numpy as np
import torch
import csv

torch.set_printoptions(edgeitems=3, threshold=30)
default_device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
torch.set_default_device(device=default_device)

wine_path = "data/p1ch4/tabular-wine/winequality-white.csv"
wine_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)

col_list = next(csv.reader(open(wine_path), delimiter=";"))
print(wine_numpy.shape, col_list)

'''
加载原始数据
'''
wineq = torch.from_numpy(wine_numpy).to(default_device)
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
target_onehot_index = target.unsqueeze(1).to(default_device)
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

# 进行单位标准差

for column in range(train_data_x.shape[1]):
    train_data_x[column] = ((train_data_x[column] - data[column].mean()) / data[column].std())

for column in range(valid_data_x.shape[1]):
    valid_data_x[column] = ((valid_data_x[column] - data[column].mean()) / data[column].std())


'''
定义模型
'''
model = torch.nn.Sequential(
    torch.nn.Linear(11, 1000),
    torch.nn.Tanh(),
    torch.nn.Linear(1000, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 10),
    torch.nn.LogSoftmax(dim=1)
)
model.to(default_device)

epoches = 500
lr = 1e-3
opt = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.NLLLoss()
batch_size = 60

# TODO 介绍 DataLoader 的使用
train_dataloader = torch.utils.data.DataLoader([(x, y) for x, y in zip(train_data_x, train_data_y)], batch_size=batch_size, shuffle=True, generator=torch.Generator(device=default_device))

total_step = 0
total_loss = 0
for i in range(epoches):
    total_loss = 0
    # TODO 进一步的说明小批量随机梯度下降
    for xs, ys in train_dataloader:
        opt.zero_grad()
        y_p = model(xs)
        # print(y_p.shape)
        # print("y_p shape", y_p.shape)
        labels = ys.argmax(dim=1).long()
        # print("labels shape", labels.shape)

        loss = loss_fn(y_p, labels)
        loss.backward()
        opt.step()

        total_step += 1
        total_loss += (loss.item() * xs.shape[0])

    if i % 10 == 0:
        print("Epoch %s step %s train/loss %.10f" % (i, total_step, loss / train_data_x.shape[0]))

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

# 实验1：epoch 1000, batch size 30, lr 1e-3, SGD, hidden neural 100
# 准确率 48.20%

# 实验1：epoch 1000, batch size 60, lr 1e-3, Adam, hidden neural 1000
# 准确率 54.25%
