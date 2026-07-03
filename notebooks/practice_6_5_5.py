'''
重点：学习 PyTorch 中的 DataLoader 和 Dataset
'''

import torch
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(100)
torch.set_default_device(device=torch.device("cpu"))

# X: 采集到的位置单位的温度
# t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]).unsqueeze(dim=1)
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]


# Y: 对应的摄氏度，目标计算的值
# t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]).unsqueeze(dim=1)
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]


def run_dataloader_example():
    # arg1 可以是几种数据类型: List，Dataset
    arg1 = [(x, y) for x, y in zip(t_u, t_c)]

    print(arg1)

    '''
    https://docs.pytorch.ac.cn/docs/2.12/data.html
    '''
    train_data = DataLoader(arg1, batch_size=3, shuffle=True, drop_last=False, generator=torch.Generator(device=torch.device("cuda:0")))

    epchos = 2

    for i in range(epchos):
        print("-----第 %s 轮-----" % i)
        batch_index = 0
        for (xs, ys) in train_data:
            print(batch_index)
            print("xs", xs)
            print("ys", ys)
            print("*" * 100)
            batch_index += 1


'''
Dataset
https://docs.pytorch.ac.cn/docs/2.12/data.html#torch.utils.data.Dataset
'''


class TempDataset(Dataset):
    '''
    温度的采集数据集
    '''

    def __init__(self, total_x, total_y):
        super().__init__()
        self.total_x = total_x
        self.total_y = total_y

    def __getitem__(self, index):
        return self.total_x[index], self.total_y[index]

    def __len__(self):
        return len(self.total_y)


def run_dataset_example():
    dataset = TempDataset(t_u, t_c)
    # print(len(dataset))
    # print(dataset[4])

    train_dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

    epchos = 2

    for i in range(epchos):
        print("-----第 %s 轮-----" % i)
        batch_index = 0
        for (xs, ys) in train_dataloader:
            print(batch_index)
            print("xs", xs)
            print("ys", ys)
            print("*" * 100)
            batch_index += 1


run_dataset_example()
