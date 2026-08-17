'''
使用 PyTorch 预训练模型 ResNet + ImageNet 实现图片的识别 | PyTorch
'''

import torch
import torchvision

'''
加载模型
'''
# 如果有 10GB 以上的内存，可以使用下面的代码
# weights = torchvision.models.ResNet152_Weights.DEFAULT
# resnet = torchvision.models.resnet152(weights=weights)

# 电脑内存，比较小，使用下面的代码
weights = torchvision.models.ResNet18_Weights.DEFAULT
resnet = torchvision.models.resnet18(weights=weights)

'''
加载数据
'''
from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        # RGB, 1-255
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# pip install pillow
from PIL import Image
import numpy as np

# img = Image.open("data/p1ch2/bobby.jpg")
img = Image.open("/home/hai/Downloads/wechat_2026-08-05_144925_414.png")

img_arr = np.asarray(img)
img_1 = img_arr[:, :, :3]
img = Image.fromarray(img_1)

img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

'''
推理
'''
resnet.eval()
out = resnet(batch_t)  # out shape 1x1000
index = torch.argmax(out.squeeze(dim=0), dim=0)
# print(index.item())

'''
得到对应的物体
分类的标签来自于：ImageNet
https://www.image-net.org/download.php
'''
labels = None
with open("data/p1ch2/imagenet_classes.txt") as fin:
    labels = [line.strip() for line in fin.readlines()]

print("识别到图片中的对象： ", labels[index.item()])
