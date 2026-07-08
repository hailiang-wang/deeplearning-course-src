#############################################
# 本节重点：在 Tensorboard 中，添加更多指标
# 先介绍混淆矩阵，查准率，查全率
# https://zhuanlan.zhihu.com/p/2036748509299397815
#############################################
import torch
from matplotlib import pyplot as plt
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(100)
torch.cuda.manual_seed_all(100)

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
data_path = "data-unversioned/p1ch7/"
cifar10 = datasets.CIFAR10(data_path, train=True, download=False)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=False)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# fig = plt.figure(figsize=(8,3))
# num_classes = 10
# for i in range(num_classes):
#     ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
#     ax.set_title(class_names[i])
#     img = next(img for img, label in cifar10 if label == i)
#     plt.imshow(img)
# plt.show()

from torchvision import transforms

# cifar10 = datasets.CIFAR10(data_path, train=True, download=False, transform=transforms.ToTensor())
# cifar10_val = datasets.CIFAR10(data_path, train=False, download=False,  transform=transforms.ToTensor())

# img_t, img_label = cifar10[0]
# print(type(img_t))
# print("img_t shape", img_t.shape) # 3x32x32
# print("img_t label", img_label)


################################
# 数据的规范化
# 比如，使用 单位标准差 方法
# https://zhuanlan.zhihu.com/p/2028540638145062215
################################
cifar10 = datasets.CIFAR10(data_path, train=True, download=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468),  # imagenet, 超過 100　万张图片上，做的统计后得到的 RGB 三个通道的均值和标准差
                         (0.2470, 0.2435, 0.2616))
]))

cifar10_val = datasets.CIFAR10(data_path, train=False, download=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468),  # imagenet, 超過 100　万张图片上，做的统计后得到的 RGB 三个通道的均值和标准差
                         (0.2470, 0.2435, 0.2616))
]))

################################
# 训练神经网络
################################

label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']  # airplane 飞机，的索引是 0, bird 的索引是 1
# 训练数据
cifar2 = [(img, label_map[label]) for img, label in cifar10 if label in [0, 2]]
# 验证数据
cifar2_val = [(img, label_map[label])
              for img, label in cifar10_val if label in [0, 2]]

print("Len of cifar2", len(cifar2))
print("Len of cifar2_val", len(cifar2_val))

# 搭建神经网络
import torch.nn as nn
import torch.optim as optim

batch_size = 10
n_epoches = 5
learning_rate = 1e-3
n_out = 2  # 希望神经的输出，是一个含有两个元素的向量，
# 比如 [0.9, 0.1]，然后约定，数值较大的索引，就是分类标签，比如 0.9 的索引是 0, 0.1 的索引是 1，那么，前面的向量代表图片属于分类 0

model = nn.Sequential(
    nn.Linear(3072, 512),  # 3072 = 32*32*3
    nn.Tanh(),
    nn.Linear(512, 32),
    nn.Tanh(),
    nn.Linear(32, n_out),
    nn.LogSoftmax(dim=1)
)
# model.to(default_device)

# 10,2 --> (10/(10+2)), (2/(10+2))
# 将使用 softmax  = 1 / 1 + e^x
opt = optim.Adam(params=model.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()

if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(
        cifar2, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=default_device))
    writer = SummaryWriter()

    total_step = 0

    for epoch in range(n_epoches):
        for imgs, labels in train_loader:
            # 20x3x32x32 -> 20x3072
            imgs = imgs.to(default_device)
            # print("imgs device", imgs.device)
            # import sys
            # sys.exit(1)

            outputs = model(imgs.view(imgs.shape[0], -1))
            loss = loss_fn(outputs, labels)

            opt.zero_grad()
            loss.backward()

            with torch.no_grad():
                opt.step()
                total_step += 1

            print(f'Step {total_step} epoch {epoch}, loss {loss}')

            writer.add_scalar("Train/Loss", loss, total_step)

    torch.save(model.state_dict(), "sample_model.pth")
    writer.close()
