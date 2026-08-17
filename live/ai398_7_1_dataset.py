'''
怎么加载数据成为 Pytorch 模型训练，所希望的，最友好的形式的数据集
https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
'''

import torch
from torch.utils.data import DataLoader, Dataset

xs = [1, 2, 3, 4, 5, 6]
ys = [1, 1, 1, 0, 0, 0]

'''
最简单的打包数据
'''
data = list(zip(xs, ys))  # [(x1, y1), (x2, y2) ...]

print(data)

data_loader = DataLoader(data, batch_size=4, shuffle=False)

# batches = list(data_loader)

# print(batches)

n = 1

for batch_x, batch_y in data_loader:
    print("第%s组数据:" % n)
    print("Xs", batch_x)
    print("Ys", batch_y)

    n += 1

'''
更灵活的，更强大的处理复杂数据的方式：Dataset
'''


class SimpleData(Dataset):

    def __init__(self):
        pass

    def __len__(self):
        '''
        返回数据集的规模，大小
        '''
        return len(xs)

    def __getitem__(self, index):
        return xs[index], ys[index] * 2


simple_data = SimpleData()
simple_loader = DataLoader(simple_data, batch_size=2, shuffle=True)

print("*" * 100)
n = 1
for batch_x, batch_y in simple_loader:
    print("第%s组数据:" % n)
    print("Xs", batch_x)
    print("Ys", batch_y)
    n += 1
