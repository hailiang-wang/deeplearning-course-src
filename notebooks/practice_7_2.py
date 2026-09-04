'''
重点：学习 PyTorch 中的 DataLoader 和 Dataset
https://docs.pytorch.ac.cn/docs/2.13/data.html
'''
import torch
from torch.utils.data import DataLoader, Dataset

data_x = [1, 2, 3, 4, 5, 6, 7, 8]
data_y = [10, 20, 30, 40, 50, 60, 70, 80]


'''
使用 DataLoader
'''
original_data = list(zip(data_x, data_y))
print(original_data)

train_data = DataLoader(original_data, batch_size=3, shuffle=True, drop_last=False)

n_epoch = 2
for i in range(n_epoch):
    print("Epoch ", i)
    for (xs, ys) in train_data:
        print("Batch -")
        print("xs", xs)
        print("ys", ys)
        print("*" * 80)

    print("----" * 20)

'''
使用 Dataset
'''


class DatasetDemo(Dataset):

    def __init__(self, xs, ys):
        super().__init__()
        self.total_x = xs
        self.total_y = ys

    def __len__(self):
        return len(self.total_x)

    def __getitem__(self, index):
        return self.total_x[index], self.total_y[index]


demo_data = DatasetDemo(xs=data_x, ys=data_y)
print("Demo data 数据集的条数：", len(demo_data))
print("Demo data 数据集的数据[1]：", demo_data[1])


train_demo_data = DataLoader(demo_data, batch_size=2, shuffle=True)

batch_index = 0
for (xs, ys) in train_demo_data:
    print("Batch index", batch_index)
    print("xs", xs)
    print("ys", ys)
    print("*" * 80)
    batch_index += 1
