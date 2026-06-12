import numpy as np
import torch
import csv

wine_path = "data/p1ch4/tabular-wine/winequality-white.csv"
wine_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)

col_list = next(csv.reader(open(wine_path), delimiter=";"))
print(wine_numpy.shape, col_list)


'''
加载原始数据

'''

wineq = torch.from_numpy(wine_numpy)
print("wineq shape", wineq.shape)
print("wineq dtype", wineq.dtype)

data = wineq[:, :-1]

target = wineq[:, -1].long()

print("data shape %s, target %s" % (data.shape, target.shape))
# 10 分类问题，0~9：0 是最低分，9 是最高分

'''
处理 Y 理想输出为独热编码 One Hot Encoding
在分类问题中，经常的使用
'''
target_onehot = torch.zeros(target.shape[0], 10)
# print("target shape", target.shape)  # [4898]
# print("target_onehot shape", target_onehot.shape)  # 4898x10

# Y
target_onehot_index = target.unsqueeze(1)
print("target_onehot_index", target_onehot_index)
print("target_onehot_index shape", target_onehot_index.shape)

target_onehot.scatter_(1, target_onehot_index, 1.0)
# print("target[0]", target[0])
# print("target_onehot[0]", target_onehot[0].tolist())


'''
数据进行规范化 -
对数据进行单位标准差
'''
data_mean = torch.mean(data, dim=0)
data_var = torch.var(data, dim=0)

# X
data_normalized = (data - data_mean) / torch.sqrt(data_var)


'''
将酒的品质，分为三个级别
'''
bad_indexes = target < 3
print("bad_indexes", bad_indexes)
print("bad_indexes shape", bad_indexes.shape)

# bad_data = target[bad_indexes]
bad_data = data[target <= 3]
mid_data = data[(target > 3) & (target < 7)]
good_data = data[target >= 7]
print("bad data size: ", bad_data.shape[0])
print("mid data size: ", mid_data.shape[0])
print("good data size: ", good_data.shape[0])

'''
分析不同级别酒的特征的平均值
'''

bad_mean = torch.mean(bad_data, dim=0)
mid_mean = torch.mean(mid_data, dim=0)
good_mean = torch.mean(good_data, dim=0)

# for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
#     print('{:2} {:20} {:6.2f}  {:6.2f}  {:6.2f}'.format(
#         i, *args))  # Python args unpacking


'''
用人为的指标，做一个区分：中等的酒的准确率
规则：
1）如果 total sulfur dioxide 的值小于 141.83，那么，这个酒，就是一个中等品质及以上的酒了。

分析一下，这个规则的准确率！
'''

total_sulfur_threshold = 141.83
total_suffur_data = data[:, 6]

# predicted_indexes = torch.lt(total_suffur_data, total_sulfur_threshold)
# [True, False, ...]
predicted_indexes = total_suffur_data < total_sulfur_threshold

# torch.Size([4898])
print("predicted_indexes shape:", predicted_indexes.shape)
print("predicted_indexes dtype:", predicted_indexes.dtype)  # torch.bool
print("predicted_indexes sum:", predicted_indexes.sum())  # tensor(2727)


# target 也是 4898 行，然后，每行都是一个数值 （0~9）；actual_indexes 就变成了 [True, False ...]
actual_indexes = target > 3
print(actual_indexes.shape, actual_indexes.dtype, actual_indexes.sum())
# torch.Size([4898]) torch.bool tensor(4878) # 真正的，不属于低品质酒的酒数量

# 我们的规则的准确率，是不是 2727/4878?
# 答案：不是。因为，我们的规则识别出的 2727,既有可能，将低品质的酒，认为是能接受的酒；也有可能是将能接受的酒，识别为了低品质的酒

n_matches = torch.sum(actual_indexes & predicted_indexes).item()
n_predirected = torch.sum(predicted_indexes).item()
n_actural = torch.sum(actual_indexes).item()

print("预测规则的准确率：%.4f" % (n_matches / n_predirected))
print("所有好酒被筛选出来的比例：%.4f" % (n_matches / n_actural))

####################################
# 我们人工的寻找规则，结果是不好的，很难！
# 机器学习，就是自动的通过给定的数据、算法、算力，去寻找规则！！！
# 在设计算法之前，需要先熟悉你的数据！！！
####################################

import pandas as pd


df = pd.read_csv(wine_path, delimiter=";")

print(df.head())
print(df.info())
print(df.describe())

print("*" * 10)
print(df.columns)

for i, row in df.iterrows():
    print("*" * 20)
    for x in df.columns:
        print("%s %s=%s" % (i, x, row[x]))
