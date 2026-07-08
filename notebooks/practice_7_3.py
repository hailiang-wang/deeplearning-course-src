# 计算混淆矩阵，查准率，查全率
# https://zhuanlan.zhihu.com/p/2036748509299397815

import sys
import torch
from matplotlib import pyplot as plt
from torchvision import datasets
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

torch.set_printoptions(edgeitems=3, threshold=20)

torch.manual_seed(100)
np.random.seed(100)
random.seed(100)

######################
# 加载数据集
# https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html
######################
data_path = "data-unversioned/p1ch7/"
# cifar10 = datasets.CIFAR10(data_path, train=True, download=False)
# cifar10_val = datasets.CIFAR10(data_path, train=False, download=False)
# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#                'dog', 'frog', 'horse', 'ship', 'truck']


# fig = plt.figure(figsize=(8, 3))
# num_classes = 10
# for i in range(num_classes):
#     ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
#     ax.set_title(class_names[i])
#     img = next(img for img, label in cifar10 if label == i)
#     plt.imshow(img)
# plt.show()

'''
本章练习题：数据增强
参考：practice_7_5.py
'''
# 1）加载数据
# 2）将飞机和鸟类过滤出来
# 3）进行分类： cifar2, cifar2_vali --> 进行数据规范化 单位标准差
# 4) 对 cifar2 进行数据增强，等到新的 cifar2_transformed
# 5）使用 cifar2_transformed 进行模型训练，使用 cifar2_vali 进行模型的评测

# cifar10 = datasets.CIFAR10(data_path, train=True, download=False) ...



'''
数据的规范化：单位标准差
'''
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468),  # imagenet, 超過 100　万张图片上，做的统计后得到的 RGB 三个通道的均值和标准差
                         (0.2470, 0.2435, 0.2616))
])
cifar10 = datasets.CIFAR10(data_path, train=True, download=False, transform=transform)

cifar10_val = datasets.CIFAR10(data_path, train=False, download=False, transform=transform)
class_names_all = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

# 过滤数据集，只需要飞机和鸟的图片
class_names = ['airplane', 'bird']
label_map = {0: 0, 2: 1}
cifar2 = [(img, label_map[label]) for img, label in cifar10 if label in [0, 2]]
cifar2_val = [(img, label_map[label]) for img, label in cifar10_val if label in [0, 2]]

# print("cifar2_val len", len(cifar2_val))
validate_inputs = torch.stack(tuple([img for (img, _) in cifar2_val]), dim=0)
validate_desired_outputs = torch.tensor([label for (_, label) in cifar2_val])
# print(validate_desired_outputs.shape)


'''
构建模型
'''
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

n_out = 2
lr = 1e-1
epoches = 10
batch_size = 10

model = nn.Sequential(
    nn.Linear(3072, 64),
    nn.Tanh(),
    nn.Linear(64, n_out),
    nn.LogSoftmax(dim=-1)
)


'''
执行训练
'''
opt = optim.SGD(model.parameters(), lr=lr)
loss_fn = nn.NLLLoss()
train_data = DataLoader(cifar2, batch_size=batch_size, shuffle=True)
total_steps = 0
writer = SummaryWriter()


for epcho in range(epoches):

    model.train()

    # 对训练集的数据进行统计
    train_loss = 0
    train_steps = 0
    train_outpus = None
    train_labels = None

    for imgs, labels in train_data:
        # print(imgs.shape)  # torch.Size([50, 3, 32, 32])
        # print(labels.shape)  # torch.Size([50])
        imgs = imgs.view(imgs.shape[0], -1)
        # print("imgs shape", imgs.shape)
        # imgs shape torch.Size([50, 3072])
        outputs = model(imgs)
        # print("outputs shape", outputs.shape)
        # outputs shape torch.Size([50, 2]) -train_outpus-> [100, 2]
        loss = loss_fn(outputs, labels)

        opt.zero_grad()
        loss.backward()

        with torch.no_grad():
            opt.step()

            total_steps += 1
            train_loss += loss.item()
            train_steps += 1

            if train_outpus is None:
                train_outpus = outputs
                train_labels = labels
            else:
                train_outpus = torch.cat((train_outpus, outputs), dim=0)
                train_labels = torch.cat((train_labels, labels), dim=0)

    # 打印本轮计算的指标
    train_loss = train_loss / train_steps
    predict_labels = torch.argmax(train_outpus, dim=-1)
    predict_correct = (predict_labels == train_labels).sum().item()

    train_accuracy = predict_correct / len(cifar2)

    with torch.no_grad():
        model.eval()

        validate_outputs = model(validate_inputs.view(validate_inputs.shape[0], -1))
        validate_predict_labels = torch.argmax(validate_outputs, dim=-1)
        validate_predict_correct = (validate_predict_labels == validate_desired_outputs).sum().item()

        validate_accuracy = validate_predict_correct / len(cifar2_val)
        validate_loss = loss_fn(validate_outputs, validate_desired_outputs).item()

        writer.add_scalars("Loss", {
            "Train": train_loss,
            "Validate": validate_loss
        }, epcho)

        writer.add_scalars("Accuracy", {
            "Train": train_accuracy,
            "Validate": validate_accuracy
        }, epcho)

        print("Epoch %s, Train/Loss %.4f, Validate/Loss %.4f, Train/Accuracy %.4f, Validate/Accuracy %.4f" % (epcho, train_loss, validate_loss, train_accuracy, validate_accuracy))


writer.close()
