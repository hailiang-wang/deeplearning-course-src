'''
基于数据增强的全连接神经网络训练
'''
import sys
import torch
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torchvision.transforms import v2
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

torch.set_printoptions(edgeitems=3, threshold=20)
torch.manual_seed(100)
np.random.seed(100)
random.seed(100)

######################
# 加载数据集
######################
data_path = "data-unversioned/p1ch7/"

# 定义标准化参数 (CIFAR-10的均值和标准差)
mean = (0.4915, 0.4823, 0.4468)
std = (0.2470, 0.2435, 0.2616)

# 基础转换：仅包含张量转换和标准化
basic_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 加载原始数据集 (不带增强)
cifar10 = datasets.CIFAR10(data_path, train=True, download=False)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=False,transform=basic_transform)

class_names_all = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

# 过滤数据集，只需要飞机和鸟的图片
class_names = ['airplane', 'bird']
label_map = {0: 0, 2: 1}
cifar2 = [(img, label_map[label]) for img, label in cifar10 if label in [0, 2]]
cifar2_val = [(img, label_map[label]) for img, label in cifar10_val if label in [0, 2]]

'''
数据增强部分
'''
augmentations = [
    v2.Compose([
        v2.RandomResizedCrop(32, scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToTensor(),
        v2.Normalize(mean=mean,std=std)
    ]),
    v2.Compose([
        v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.ToTensor(),
        v2.Normalize(mean=mean,std=std)
    ]),
    v2.Compose([
        transforms.ColorJitter(hue=0.1),
        v2.ToTensor(),
        v2.Normalize(mean=mean,std=std)
    ]),
    v2.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10),
        v2.ToTensor(),
        v2.Normalize(mean=mean,std=std)
    ]),
    transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.7, 1.0)),
        transforms.ColorJitter(saturation=0.3),
        v2.ToTensor(),
        v2.Normalize(mean=mean,std=std)
    ])
]

# 应用增强：对cifar2中的每张图片生成5张增强后的图片
cifar2_transformed = []

for img, label in cifar2:
    for i in range(5):
        augmented_img = augmentations[i](img)
        cifar2_transformed.append((augmented_img, label))


# 准备验证集张量
validate_inputs = torch.stack(tuple([img for (img, _) in cifar2_val]), dim=0)
validate_desired_outputs = torch.tensor([label for (_, label) in cifar2_val])

'''
构建模型
'''
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

n_out = 2
lr = 1e-3
epochs = 10  
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
train_data = DataLoader(cifar2_transformed, batch_size=batch_size, shuffle=True)  # 使用增强后的数据
total_steps = 0
writer = SummaryWriter()

for epoch in range(epochs):
    model.train()
    
    # 训练统计
    train_loss = 0
    train_steps = 0
    
    for imgs, labels in train_data:
        imgs = imgs.view(imgs.shape[0], -1)
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_steps += 1
        train_loss += loss.item()
        train_steps += 1

    # 验证阶段
    with torch.no_grad():

        # 训练集准确率计算（使用当前epoch的输出）
        train_outputs = model(validate_inputs.view(validate_inputs.shape[0], -1))  # 重用验证集输入进行评估
        train_predict_labels = torch.argmax(train_outputs, dim=-1)
        train_accuracy = (train_predict_labels == validate_desired_outputs).sum().item() / len(cifar2_val)


        model.eval()
        validate_outputs = model(validate_inputs.view(validate_inputs.shape[0], -1))
        validate_predict_labels = torch.argmax(validate_outputs, dim=-1)
        validate_predict_correct = (validate_predict_labels == validate_desired_outputs).sum().item()
        
        validate_accuracy = validate_predict_correct / len(cifar2_val)
        validate_loss = loss_fn(validate_outputs, validate_desired_outputs).item()

        
        writer.add_scalars("Loss", {
            "Train": train_loss / train_steps,
            "Validate": validate_loss
        }, epoch)

        writer.add_scalars("Accuracy", {
            "Train": train_accuracy,
            "Validate": validate_accuracy
        }, epoch)

        print(f"Epoch {epoch}, Train/Loss {train_loss/train_steps:.4f}, "
              f"Validate/Loss {validate_loss:.4f}, Train/Accuracy {train_accuracy:.4f}, "
              f"Validate/Accuracy {validate_accuracy:.4f}")

writer.close()