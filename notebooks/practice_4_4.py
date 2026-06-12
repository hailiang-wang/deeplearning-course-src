import numpy as np
import torch

torch.set_printoptions(edgeitems=3, threshold=30)


'''
读取数据
'''
bike_numpy = np.loadtxt("data/p1ch4/bike-sharing-dataset/hour-fixed.csv",
                        dtype=np.float32,
                        delimiter=",",
                        skiprows=1,
                        converters={1: lambda x: float(x[8:10])})

bikes = torch.from_numpy(bike_numpy)

import pandas as pd

pd.DataFrame(bikes).to_csv("bikes_converters.csv", index=False)

print("bike.shape", bikes.shape)  # 17520x17
print("bike.stride", bikes.stride())

'''
将数据进行转化，成为 天数 x 特征数 x 24小时
得到：daily_bikes_t
'''
daily_bikes = bikes.view(-1, 24, bikes.shape[1])

print(daily_bikes.shape)

daily_bikes_t = daily_bikes.transpose(dim0=2, dim1=1)
print("daily_bikes_t", daily_bikes_t.shape)  # 天数x特征数x24

print("24 value of 0 day temp", daily_bikes_t[0][10].tolist())


'''
将 weathersit 转化为 onehot 编码的数据
'''
# 以第一天的数据为例子
first_day = bikes[:24].long()
print("first_day", first_day.shape)  # 24x17
weather_onehot = torch.zeros(first_day.shape[0], 4)
weather_class = first_day[:, 9]
print("weather_class shape", str(weather_class.shape))
weather_onehot.scatter_(-1, weather_class.unsqueeze(1) - 1, 1)
print("weather_onehot shape", weather_onehot.shape)
# print("weather_onehot", weather_onehot.tolist())

'''
将 weathersit onehot，汇总到 bike 信息中
'''
# cat 拼接，stack 堆叠
firstday_with_classify_wf = torch.cat((first_day, weather_onehot), dim=-1)
print("firstday_with_classify_wf",
      firstday_with_classify_wf.shape)  # 24x17 24x4 --> ? 24 x (17+4)
# 思考：之前 stack 的应用场景是什么？stack 和 cat 的区别？
# 1. cat 要求，输入的张量，都有相同的维度数，并且要求非拼接的那个维度数必须一样
# 2. cat 的运算结果，并不会增加新的维度
# 3. stack 要求，输入的张量，shape 是一致
# 4. stack 生成张量，会增加一个维度
# 具体的使用介绍，https://zhuanlan.zhihu.com/p/2032429599104218598


'''
将 bikes 中，所有的 weathersit 转化为 onehot 向量，然后拼接到 bikes 信息中
'''
days_weather_onehot = torch.zeros(daily_bikes.shape[0], 4, 24)
days_weather_onehot.scatter_(dim=1,
                             index=daily_bikes_t[:, 9, :].long().unsqueeze(1) - 1, value=1.0)
daily_bikes_t = torch.cat(
    (daily_bikes_t, days_weather_onehot), dim=1)  # [730, 21, 24]

import pandas
pandas.DataFrame(daily_bikes_t[0]).to_csv("daily_bikes_t0.csv", index=False)
print("daily_bikes_t.shape", daily_bikes_t.shape)
