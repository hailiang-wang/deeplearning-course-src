#############################################
# 本节重点：使用迁移学习，训练模型
# tensorboard.exe --logdir ./runs/
#############################################
import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))

import torch
from matplotlib import pyplot as plt
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import random
import numpy as np

torch.manual_seed(100)
np.random.seed(100)
random.seed(100)

################################
# 设置默认的设备，有 GPU 的话，默认使用 GPU
################################
default_device = torch.device("cpu")
if torch.cuda.is_available():
    torch.set_default_device(torch.device("cuda"))
    default_device = torch.device("cuda")


print("default device", default_device)

################################
# 加载数据集
################################

data_path = "data-unversioned/p1ch7"
cifar10 = datasets.CIFAR10(data_path, train=True, download=False)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=False)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

from torchvision import transforms

################################
# 数据的规范化
# 比如，使用 单位标准差 方法
# https://zhuanlan.zhihu.com/p/2028540638145062215
################################
cifar10 = datasets.CIFAR10(data_path, train=True, download=False, transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468),  # imagenet, 超過 100　万张图片上，做的统计后得到的 RGB 三个通道的均值和标准差
                         (0.2470, 0.2435, 0.2616))
]))

cifar10_val = datasets.CIFAR10(data_path, train=False, download=False, transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468),  # imagenet, 超過 100　万张图片上，做的统计后得到的 RGB 三个通道的均值和标准差
                         (0.2470, 0.2435, 0.2616))
]))

################################
# 分割出 2 分类的数据集
################################
print("Load dataset ...")
label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']  # airplane 飞机，的索引是 0, bird 的索引是 1
# 训练数据
cifar2 = [(img, label_map[label]) for img, label in cifar10 if label in [0, 2]]
# 验证数据
cifar2_val = [(img, label_map[label])
              for img, label in cifar10_val if label in [0, 2]]

print("  Len of cifar2", len(cifar2))
print("  Len of cifar2_val", len(cifar2_val))

validate_inputs = torch.stack(
    tuple([img.to(default_device) for (img, _) in cifar2_val]), dim=0)
validate_labels = torch.tensor([label for (_, label) in cifar2_val])

'''
搭建迁移学习网络
'''
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models as PRETRAINED_MODELS

batch_size = 30
n_epoches = 10
learning_rate = 1e-3
n_out = 2

resnet_weights = PRETRAINED_MODELS.ResNet18_Weights.DEFAULT
resnet_model = PRETRAINED_MODELS.resnet18(resnet_weights)


def add_custom_layers(model: nn.Module, out_number: int):
    '''
    将给定的神经网络，进行续接：增加新的层，完成新的任务
    '''

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features

    custom_layers = nn.Sequential(
        nn.Linear(num_features, 32),
        nn.ReLU(),
        nn.Linear(32, out_number),
        nn.LogSoftmax(dim=-1)
    )

    model.fc = custom_layers

    return model


model = add_custom_layers(resnet_model, n_out)

# 10,2 --> (10/(10+2)), (2/(10+2))
# 将使用 softmax  = 1 / 1 + e^x
opt = optim.Adam(params=model.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()

if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(
        cifar2, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=default_device))
    writer = SummaryWriter()

    total_step = 0

    print("Start to train neural network ...")
    for epoch in range(n_epoches):
        model.train()
        # 对训练的 Loss 进行记录
        train_loss = 0
        train_step = 0
        train_outputs = None
        train_labels = None

        for imgs, labels in train_loader:
            # 20x3x32x32 -> 20x3072
            imgs = imgs.to(default_device)
            # outputs = model(imgs.view(imgs.shape[0], -1))
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            opt.zero_grad()
            loss.backward()

            with torch.no_grad():
                opt.step()
                train_loss += loss.item()
                train_step += 1
                total_step += 1

                if train_outputs is None:
                    train_outputs = outputs
                    train_labels = labels
                else:
                    train_outputs = torch.cat((train_outputs, outputs), dim=0)
                    train_labels = torch.cat((train_labels, labels), dim=0)

            # print(f'Step {total_step} epoch {epoch}, loss {loss}')

        '''
        每 1 个 Epoch 完成训练后，进行评测
        '''
        # 记录训练的损失函数值
        train_loss = train_loss / train_step

        # 记录训练的准确率
        predict_labels = torch.argmax(train_outputs, dim=-1)
        predict_correct = (predict_labels == train_labels).sum()
        train_accuracy = predict_correct / predict_labels.numel()

        # 进行验证集数据的预测
        with torch.no_grad():
            model.eval()
            # 计算验证集上的损失
            validate_output = model(validate_inputs)

            validate_loss = loss_fn(validate_output, validate_labels)

            # 计算验证集上的准确率
            predict_labels = torch.argmax(validate_output, dim=-1)
            predict_correct = (predict_labels == validate_labels).sum()
            validate_accuracy = predict_correct / len(cifar2_val)

            writer.add_scalars("Loss", {
                "Train": train_loss,
                "validate": validate_loss.item()
            }, epoch)

            writer.add_scalars("Accuracy", {
                "Train": train_accuracy.item(),
                "validate": validate_accuracy.item()
            }, epoch)

            print("  Epoch %s, train (loss %.4f, accuracy %.4f), validate(loss %.4f, accuracy %.4f)" %
                  (epoch, train_loss, train_accuracy.item(), validate_loss.item(), validate_accuracy.item()))

    print("Train done, model saved to sample_model.pth, checkout log in ./runs with tensorboard.")
    torch.save(model.state_dict(), "sample_model.pth")
    writer.close()
