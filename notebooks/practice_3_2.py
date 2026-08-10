'''
使用 Pytorch 创建张量
'''
import torch

# 打印的样式
torch.set_printoptions(edgeitems=10, threshold=50)

x = torch.randn(20, 1024)
print(x)
print(x.shape)

# Max 运算
x = torch.randn(3, 3)
values, indices = torch.max(x, dim=1)

print(x)
print("values", values)
print("indices", indices)

# ArgMax 运算
y = torch.argmax(x, dim=1)
print(y)

'''
Slicing / 分片操作
'''
x = torch.randn(3, 3, 4)
print(x)

from copy import deepcopy

z = x[0:2, :2, :2]
y = torch.ones_like(z)
y[:, :, :] = z

print("y", y)
print(y.shape)
y[0, 0, 0] = 1
print(x[0, 0, 0])

print("*" * 80)
print(x)
print("*" * 80)
print(y)

# x[:, 1, 0] = 0
# print(x)

print("x" * 80)
x0 = x[0, 0, 0]
print(id(x0))  # <5>
print(id(x[0, 0, 0]))  # <2>
print(x[0, 0, 0])  # <3>

print("y" * 80)
y0 = y[0, 0, 0]
print(id(y0))  # <6>
print(id(y[0, 0, 0]))  # <1>
print(y[0, 0, 0])  # <4>

# 以上代码是违反直觉，因为 <1> 和 <2> 得到的 id 是相同的值，但是 <3> 和 <4> 的值是不一样的。

# 解释：
#  pytorch 为了提升效率，对于张量使用了内存池，id 函数调用时，是返回了一个最新的复制出来的临时张量
# 如果想要获得一个张量的底层的真实的地址，需要使用
print("x", id(x.storage().data_ptr()))  # <7>
print("y", id(y.storage().data_ptr()))  # <8>
print("z", id(z.storage().data_ptr()))  # <9>
# print(z.data)

# 补充：
# 以上 <5> 和 <6> 打印出来的 id 值是不同的。
# <7> 和 <9> 的值是相同的

'''
节省内存
'''
x = torch.randn(3, 2)
print(id(x))
y = torch.ones_like(x)
x[:] = x + y
print(x)
# ... 后面的代码中，不需要再使用 x 了
print(id(x))

'''
和 numpy 的转换
'''
import numpy as np

x = np.array([[1, 2], [3, 4]])
print(x)
print(x.shape)

y = torch.from_numpy(x)
print(y)
print(y.shape)

z = y.numpy()
print(id(x.data))
print(id(y.storage().data_ptr()))
