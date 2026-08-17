'''
使用 Tensorboard 来查看更多指标： train loss, 准确率； validate loss， 准确率
pip install tensorboard
'''

import torch
from matplotlib import pyplot as plt
from torchvision import datasets, transforms as T
import numpy as np
from PIL import Image
import sys
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(100)
np.random.seed(100)

########################
# 加载数据集
########################
data_path = "data-unversioned/p1ch7"
# 训练集
cifar10 = datasets.CIFAR10(data_path, train=True, download=False)
# 验证集
cifar10_vali = datasets.CIFAR10(data_path, train=False, download=False)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# fig = plt.figure(figsize=(8, 3))
# num_classes = 10
# for i in range(num_classes):
#     ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
#     ax.set_title(class_names[i])
#     img = next(img for img, label in cifar10 if label == i)
#     print("照片：", class_names[i])
#     img_arr = np.asarray(img)
#     print(img_arr.shape)
#     # 照片： horse
#     # (32, 32, 3) 高x宽xRGB通道
#     print(img_arr)
#     plt.imshow(img)
#     # （32x32x3） - 机器学习习惯使用 通道x宽x高 ->
#     img_t = torch.permute(torch.from_numpy(img_arr), (2, 1, 0))
#     print("img_t", img_t.shape)
# plt.show()

# 分割数据，只需要鸟类和飞机的照片
label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']  # 飞机的索引 0,鸟类的索引是 1
transforms = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4915, 0.4823, 0.4468),  # imagenet, 超過 100　万张图片上，做的统计后得到的 RGB 三个通道的均值和标准差
                (0.2470, 0.2435, 0.2616))
])

cifar2 = [(transforms(img), label_map[label]) for img, label in cifar10 if label in [0, 2]]
# for (img, label) in cifar2:
#     plt.cla()
#     plt.imshow(img)
#     plt.show()
cifar2_val = [(transforms(img), label_map[label]) for img, label in cifar10_vali if label in [0, 2]]
print("cifar2 训练数据集 size", len(cifar2))
print("cifar2 训练验证集 size", len(cifar2_val))

cifar2_val_inputs = torch.stack(tuple([img for img, _ in cifar2_val]), dim=0)
cifar2_val_labels = torch.tensor([label for _, label in cifar2_val])

#########################################
# 搭建神经网络
#########################################
import torch.nn as nn
import torch.optim as optim

batch_size = 30
n_epchoes = 10
lr = 1e-3
n_out = 2

# 3x32x32
model = nn.Sequential(
    nn.Linear(3 * 32 * 32, 4196),  # 3x32x32 = 3072
    nn.Tanh(),
    nn.Linear(4196, 512),
    nn.Tanh(),
    nn.Linear(512, 64),
    nn.Tanh(),
    nn.Linear(64, n_out),
    nn.LogSoftmax(dim=-1)
)

opt = optim.SGD(params=model.parameters(), lr=lr)
loss_fn = nn.NLLLoss()

if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(cifar2, batch_size=batch_size, shuffle=True)

    writer = SummaryWriter()

    total_step = 0

    for epoch in range(n_epchoes):
        train_steps = 0
        train_loss = 0
        train_labels = None
        train_outputs = None

        model.train()
        for imgs, labels in train_loader:
            # print(imgs.shape)  # 30x3x32x32
            # print(labels.shape)  # 30
            outputs = model(imgs.view(imgs.shape[0], -1))
            # print("outputs shape", outputs.shape)  # 30x2
            loss = loss_fn(outputs, labels)  # outputs 30x2, labels 30
            # output1 [0.1, 0.9] --> 属于分类1 ： 0 --> 分类错误了！
            # 损失：通过预测的索引，与理想的索引，进行差异的计算

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_step += 1
            train_steps += 1
            train_loss += loss.item()
            if train_labels is None:
                train_labels = labels
                train_outputs = outputs
            else:
                train_labels = torch.cat((train_labels, labels), dim=0)
                train_outputs = torch.cat((train_outputs, outputs), dim=0)

        # 一个 epoch 结束，打印更多的信息
        train_loss = train_loss / train_steps  # 训练集损失函数
        predict_labels = torch.argmax(train_outputs, dim=-1)
        predict_correct = (predict_labels == train_labels).sum().item()

        train_accuracy = predict_correct / len(cifar2)  # 训练集准确率

        with torch.no_grad():
            model.eval()
            # cifar2_val_inputs 2000x3x32x32 --> 2000x3072
            validate_outputs = model(cifar2_val_inputs.view(len(cifar2_val), -1))
            validate_loss = loss_fn(validate_outputs, cifar2_val_labels).item()

            validate_correct = (torch.argmax(validate_outputs, dim=-1) == cifar2_val_labels).sum().item()
            validate_accuracy = validate_correct / len(cifar2_val)

            writer.add_scalars("Loss", {
                "Train": train_loss,
                "vali": validate_loss
            }, epoch)

            writer.add_scalars("Accuracy", {
                "Train": train_accuracy,
                "vali": validate_accuracy
            }, epoch)

            print("Epoch %s, train loss %.4f, train accuracy %.4f, validate loss %.4f, validate accuracy %.4f" % (epoch, train_loss, train_accuracy, validate_loss, validate_accuracy))

    torch.save(model.state_dict(), "sample_model.pth")
    writer.close()
