import numpy as np
import torch
import torch.optim as optimimzer
import torch.nn as nn
import sys
import random
import numpy as np
import csv
from torchinfo import summary
torch.set_printoptions(edgeitems=2, precision=2, threshold=20)

torch.manual_seed(100)
random.seed(100)
np.random.seed(100)

wine_path = "data/p1ch4/tabular-wine/winequality-white.csv"
wine_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)

wineq = torch.from_numpy(wine_numpy)

data = wineq[:, :-1]
target = wineq[:, -1].long()

target_onehot = torch.zeros(target.shape[0], 10)
tu = target.unsqueeze(1)
target_onehot.scatter_(1, tu, 1.0)

data_mean = torch.mean(data, dim=0)
data_std = torch.std(data, dim=0)
data_norm = (data - data_mean) / data_std

n_samples = data_norm.shape[0]
n_validate = int(0.2 * n_samples)

shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_validate]
validate_indices = shuffled_indices[-n_validate:]

train_data_x = data_norm[train_indices]
train_data_y = target_onehot[train_indices]

validate_data_x = data_norm[validate_indices]
validate_data_y = target_onehot[validate_indices]

model = torch.nn.Sequential(
    nn.Linear(11, 1024),
    nn.Tanh(),
    nn.Linear(1024, 512),
    nn.Tanh(),
    nn.Linear(512, 64),
    nn.Tanh(),
    nn.Linear(64, 10),
    nn.LogSoftmax(dim=-1))
summary(model, depth=2)

loss_fn = nn.NLLLoss()

learning_rate = 1e-2
opt = optimimzer.SGD(model.parameters(), lr=learning_rate)
epochs = 20000

for epoch in range(1, epochs + 1):
    t_p = model(train_data_x)

    # https://docs.pytorch.org/docs/2.13/generated/torch.nn.NLLLoss.html
    t_y = train_data_y.argmax(dim=-1)
    # t_p shape 3919x10, 比如第一行 是 [0,0,1,0,0,0,0,0 ...] 它是哪个分类？ 2
    # t_y vector (3919) [2, ...]
    loss = loss_fn(t_p, t_y)

    opt.zero_grad()
    loss.backward()

    opt.step()
    print('Epoch %d, Train Loss %f' % (epoch, float(loss)))

    if not torch.isfinite(loss).all():
        break

    if epoch % 100 == 0:
        with torch.no_grad():
            valid_loss = loss_fn(model(validate_data_x), validate_data_y.argmax(dim=-1))
            print('Epoch %d, Validate Loss %f' % (epoch, float(valid_loss)))

yz = model(validate_data_x)
print("yz shape", yz.shape)
yz_a = (torch.argmax(yz, dim=-1))
print(yz_a)

number_yz = (yz_a == target[validate_indices]).sum().item()
print("预测准确的酒数量:", number_yz)

total = len(validate_data_y)
print("一共预测了多少种酒:", total)

new_accuracy = number_yz / total
print(new_accuracy)
