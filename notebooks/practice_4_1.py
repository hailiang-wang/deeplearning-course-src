import torch
import imageio

img_arr = imageio.imread("./data/p1ch4/image-dog/bobby.jpg")
# print(img_arr.shape)  # (720 高, 1280 宽, 3 通道)
# print(type(img_arr))

# 3x1280x720
img = torch.from_numpy(img_arr)
# print("img shape", img.shape)
img = img.permute(2, 1, 0)
# print("img shape", img.shape)  # 3 channel, 1280 width, 720 height

'''
批量的加载图片到一个 batch
'''
import os

data_dir = './data/p1ch4/image-cats/'
filenames = [name for name in os.listdir(data_dir)
             if os.path.splitext(name)[-1] == '.png']
batch = []

for i, filename in enumerate(filenames):
    img_arr = imageio.imread(os.path.join(data_dir, filename))
    img_t = torch.from_numpy(img_arr)
    img_t = img_t.permute(2, 1, 0)
    img_t = img_t[:3]  # <1>
    batch.append(img_t)

batch = torch.stack(tuple(batch), dim=0)
# 3x3x256x256
# print(batch.shape)
# print(batch)

'''
数据规范化：使用单位标准差的方法
https://zhuanlan.zhihu.com/p/2028540638145062215
'''
# data = torch.tensor([[1, 65], [3, 130], [2, 80], [2, 70],
#                     [1, 50]], dtype=torch.float32)
# print(f'mean {data.mean(dim=0)}, std {data.std(dim=0)}')

# data_t = (data - data.mean(dim=0)) / data.std(dim=0)
# print(data_t)
# batch = batch.float()
batch = batch / 255.0

n_channels = batch.shape[1]
for c in range(n_channels):
    mean = torch.mean(batch[:, c])  # 0 R, 1 G, 2 B
    std = torch.std(batch[:, c])
    batch[:, c] = (batch[:, c] - mean) / std

print(batch)
print(batch.shape)
print(batch.dtype)
