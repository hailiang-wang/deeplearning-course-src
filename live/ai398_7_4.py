'''
使用 torchvision 进行图片的变换

## 官方文档

torchvision==0.25.0+cpu
https://docs.pytorch.org/vision/0.22/transforms.html

## 使用介绍
https://zhuanlan.zhihu.com/p/2084218976

'''

import torch
from matplotlib import pyplot as plt
from torchvision import datasets
from torchvision.transforms import v2 as T
from PIL import Image
import numpy as np

data_path = "data-unversioned/p1ch7/"
cifar10 = datasets.CIFAR10(data_path, train=True, download=False)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=False)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

fig = plt.figure(figsize=(8, 3))
num_classes = 10
original_images = []

for i in range(num_classes):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    ax.set_title(class_names[i])
    img = next(img for img, label in cifar10 if label == i)
    plt.imshow(img)
    original_images.append(img)

plt.savefig("tmp_7_5_original.png", format="png")
plt.clf()


transformed_imgs = []

transforms = T.Compose([
    T.RandomResizedCrop(size=(32, 32), antialias=True),
    T.RandomHorizontalFlip(p=0.9),
    T.ColorJitter(brightness=(0.5, 1.5)),
])

for img in original_images:
    img_t = torch.from_numpy(np.asarray(img)).permute(2, 1, 0)
    img_t: torch.Tensor = transforms(img_t)
    img = Image.fromarray(img_t.permute(2, 1, 0).numpy())
    transformed_imgs.append(img)

fig2 = plt.figure(figsize=(8, 3))
for i in range(num_classes):
    ax = fig2.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    ax.set_title(class_names[i])
    plt.imshow(transformed_imgs[i])

plt.savefig("tmp_7_5_post.png", format="png")

'''
练习题：

调整 ai398_7_3.py 代码，使用 torchvision 进行图片数据增强：
1）对飞机和小鸟的图片，每张图片各生成 5 张新的图片，比如使用随机缩放并剪切的方式生成一张，采用 RandomAffine 生成一张等
2）生成后，每张新的图片还是 3x32x32

然后，进行模型的训练。
分析模型的准确率变化了吗？是提升了，还是减小了？