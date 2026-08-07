'''
使用 Pytorch 创建张量
'''
import torch

# 创建一个行向量
x: torch.Tensor = torch.arange(12)
# x = torch.arange(4, 12)
print(x)
print(x.shape)
print(x.numel())

# 使用 reshape 创建一个多维度的 tensor
y = x.reshape((2, 6))
print(y.shape)
print(y)

z = x.reshape((2, 2, -1))
print(z.shape)
print(z)

# Method 2
x = torch.zeros((3, 3, 2))
print(x)
print(x.shape)

x = torch.ones((3, 3, 2))
print(x)
print(x.shape)

# Method 3
x = torch.randn(3, 3, 2)
print(x)
print(x.shape)

# Method 4
y = torch.ones_like(x)
y = torch.zeros_like(x)

# Method 5
x = [1, 1.0, 0]
y = torch.tensor(x)
print(y.shape)
print(y)

x = [[1, 1.1, 0], [2, 1.0, 0]]  # shape 2x3
y = torch.tensor(x)
print(y.shape)
print(y.dtype)
print(y)

# Dtype 的转化
z = y.to(torch.int8)
print("z", z)
print(z.dtype)

# 数据的转化，有代价，可能造成小数部分的丢失
print(z.to(torch.float32))

'''
张量运算
'''
x = torch.tensor([1.0, 2, 4, 8, 9])  # 1x5
y = 2

print("加法运算：")
print(x)
z = y + x  # 使用广播的形式，完成计算
# print(z)

y = torch.randn(2, 5, 1)
print(y)
z = x + y  # z shape
print(z.shape)  # 2x5x5
print(z)

'''
几个运算的函数
'''
# e 指数运算
y = torch.exp(x)
print(y)
print(y.shape)

# concatenate 拼接： 原则
# 1） 待拼接的张量们，都有相同的轴数
# 2） 待拼接的张量们，在拼接的轴上，元素数可以不一样，也可以一样
# 3） 待拼接的张量们，在非拼接的轴上，元素数，必须是一样的
print("拼接")
x = torch.randn(3, 2)  # x,y 都有两个轴
y = torch.randn(3, 1)
z = torch.cat([x, y], dim=1)
print(x)
print(y)
print(z)

'''
二元运算
'''

x = torch.tensor([1, 2, 3, 4])
y = torch.tensor([0, 2, 4, 4])
z = (x > y)
print(z.shape)
print(z.dtype)
print(z)  # tensor([ True, False, False, False])
print(z.sum())  # tensor(1)
print(z.sum().item())  # 1

'''
求和运算
'''
x = torch.randn((3, 2, 3), dtype=torch.float32)
y = x.sum(dim=1)
print(x)
print(y.shape)
print(y)


'''
嵌套操作 
squeeze （解套） / unsqueeze （加套）
'''
# unsqueeze
x = torch.randn(3, 3, 1, 1)
y = torch.unsqueeze(x, dim=1)
print(x)
print(y)
print(y.shape)

# squeeze
print("*" * 80)
y = torch.squeeze(x, dim=-2)
print(x)
print(y)
print(y.shape)

x = torch.randn(2, 2)
y = torch.unsqueeze(x, dim=-1)
print(y.shape)  # 2x2x1
y = torch.unsqueeze(x, dim=1)  # dim N 等于多少，就是在完成加套后，新加在第N轴
print(y.shape)  # 2x1x2

'''
Softmax
缓慢的变大

1) 和是1
2）可以看成是概率分布
3）x 值越大， y 变化越平滑，但是依然是大
'''
print("*" * 80)
x = torch.randn(2, 2)
y = torch.softmax(x, dim=0)
print(x)
print(y.shape)
print(y)

# x tensor([[-0.4756, -0.8313],
#         [-1.4213,  0.1071]])
# y shape: torch.Size([2, 2])
# y tensor([[0.5880, 0.4120],
#         [0.1782, 0.8218]])

x1 = torch.exp(x[0][0]).item()
x2 = torch.exp(x[1][0]).item()
x_sum = x1 + x2

print("x[0][0] -->", x1 / x_sum)
print("x[1][0] -->", x2 / x_sum)
