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
from torchvision.transforms import v2
from PIL import Image
'''
pip install torchvision==0.25.0+cpu
'''
import numpy as np

data_path = "data-unversioned/p1ch7/"
cifar10 = datasets.CIFAR10(data_path, train=True, download=False)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=False)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

fig = plt.figure(figsize=(8,3))

num_classes = 10
original_images = []

for i in range(num_classes):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    ax.set_title(class_names[i])
    img: Image = next(img for img, label in cifar10 if label == i)
    plt.imshow(img)
    original_images.append(img)

plt.savefig("tmp_7_5_original.png", format="png")
plt.clf()

'''
数据增强
'''

transformed_imgs = []
transforms = v2.Compose([
    # v2.RandomResizedCrop((32,32), antialias=True),
    # v2.RandomHorizontalFlip(p=1.0)
    # v2.RandomAffine((30,90))
    v2.ColorJitter(brightness=0.5, hue=0.3),
])

for img in original_images:
    img_t: torch.Tensor = torch.from_numpy(np.asarray(img)).permute(2,1,0)
    print("img_t shape", img_t.shape)
    img_t = transforms(img_t)
    transformed_img = Image.fromarray(img_t.permute(2,1,0).numpy())
    transformed_imgs.append(transformed_img)

fig2 = plt.figure(figsize=(8,3))
for i in range(num_classes):
    ax = fig2.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    ax.set_title(class_names[i])
    plt.imshow(transformed_imgs[i])

plt.savefig("tmp_7_5_post.png", format="png")








