'''
图像分类：实现鸟类和飞机的分类
* 使用 Tensorboard 查看训练的过程
'''
import sys
import torch
from torch import Tensor
from matplotlib import pyplot as plt
from torchvision import datasets
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

# 在寻找最佳的超参数和网络的过程中，使用的随机数
# 种子；在真正的全量数据上，注释掉
torch.manual_seed(100)

default_device = torch.device("cpu")
if torch.cuda.is_available():
    default_device = torch.device("cuda:0")

torch.set_default_device(device=default_device)
print("Default device:", default_device)

'''
加载数据集
'''
data_path = "data-unversioned/p1ch7/"
cifar10 = datasets.CIFAR10(data_path, train=True, download=False)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=False)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# fig = plt.figure(figsize=(8, 3))
# num_classes = 10
# for i in range(num_classes):
#     ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
#     ax.set_title(class_names[i])
#     img = next(img for img, label in cifar10 if label == i)
#     plt.imshow(img)
# plt.show()

from torchvision import transforms
from torchvision.transforms import Compose

preprocess = Compose([
    transforms.ToTensor(),
    transforms.Resize([32, 32]),
    # 做的统计后得到的 RGB 三个通道的均值和标准差
    # imagenet, 超過 100　万张图片上
    # https://www.image-net.org/download.php
    transforms.Normalize((0.4915, 0.4823, 0.4468),  # mean
                         (0.2470, 0.2435, 0.2616)),  # std
])

cifar10 = datasets.CIFAR10(data_path, train=True, download=False, transform=preprocess)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=False, transform=preprocess)

'''
制作新的数据集：cifar2
'''
label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']  # airplane 飞机，的索引是 0, bird 的索引是 1
cifar2 = [(img, label_map[label]) for img, label in cifar10 if label in [0, 2]]
cifar2_val = [(img, label_map[label]) for img, label in cifar10_val if label in [0, 2]]

print("Cifar2 数据集：训练集(%s), 验证集(%s)" % (len(cifar2), len(cifar2_val)))

'''
搭建神经网络模型
'''
import torch.nn as nn
import torch.optim as optimizer

# 超参数， hyper params
# 是因为，我们使用的方法，全称: 小批量随机梯度下降（mini-batch SGD）
# 使用习惯, SGD：全量 1000 图片 = 50 个小数据集 x 20 个图片作为一批
batch_size = 20  # 一次性输入 20 张图片，然后累积 loss，再进行梯度计算和参数更新

# 有关疑问：
# 1. 如果每次训练，都是用 1 张图片，计算 loss，那么这个方法不行：1）振荡不稳定，2）效率低，慢
# 2. 如果每次训练，都用全量去更新，1000 张图片，这个方法，也不行：更新速度慢，学习的慢
# 3. SGD 之所以工作，就是因为，考虑了不同数据之间的差异，做了平衡

# 小批量随机梯度下降 的过程：
# 1) 设定 batch size; 2) 设定训练的轮数 e.g. 10
# 每一轮，都将全量数据打乱，然后分成小批量的子数据集

n_epochs = 10
learning_rate = 1e-3
n_out = 2

model = nn.Sequential(
    nn.Linear(3 * 32 * 32, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, n_out),
)
model.to(device=default_device)
summary(model)

opt = optimizer.SGD(params=model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

if __name__ == "__main__":

    writer = SummaryWriter()
    train_dataloader = torch.utils.data.DataLoader(
        cifar2, batch_size=batch_size, shuffle=True,
        generator=torch.Generator(device=default_device)
    )

    total_step = 0

    for epoch in range(1, n_epochs + 1):
        current_loss = 0

        for imgs, labels in train_dataloader:
            # imgs shape batchx3x32x32
            imgs = imgs.to(default_device)
            flatten_imgs = imgs.view(imgs.shape[0], -1)
            # flatten_imgs shape batchx3072
            predicts = model(flatten_imgs)  # 20x2 = [[0.1, 0.8], ... ]
            # predicts
            # print(predicts.shape)  # batch_sizex2

            # print(labels.shape)  # batch_size
            # torch.Size([20]), 20 是 batch_size
            # tensor([1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1])

            labels_onehot = torch.zeros_like(predicts)
            labels_onehot = labels_onehot.scatter(1, labels.unsqueeze(dim=1), 1.0).to(default_device)
            # tensor([[0,1], [0,1], [0,1], [1,0], ...])
            # print(labels_onehot)
            # print(labels_onehot.shape)

            loss = loss_fn(predicts, labels_onehot)

            opt.zero_grad()
            loss.backward()

            with torch.no_grad():
                opt.step()
                total_step += 1

            print("Epoch %s, Total step %s, Loss %.4f" % (epoch, total_step, loss.item()))
            writer.add_scalar("Train/Loss", loss.item(), total_step)

    torch.save(model.state_dict(), "ai501_7_1.pth")
    writer.close()
