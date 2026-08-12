'''
在 GPU 机器上，创建和进行张量运算
在 GPU 上的，管理 PyTorch 张量，有几种方法
'''

import torch
from datetime import datetime

# 方法1
device = torch.device("cuda:0")
x = torch.randn(2, 3, device=device)
print(x)

y = torch.randn(3, 2)
print(y)

# 如果两个张量，不再同一个类型的设备上，计算会出错
# z = torch.mm(x, y)

# 方法2
z = y.to(device=device)

u = torch.mm(x, z)
print(u.shape)

# 方法3
torch.set_default_device(device=device)
x = torch.randn(2, 3)
print(x)

# 常用的兼容性代码的写法
default_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_device(default_device)

'''
对比 GPU 和 CPU 运算的速度
'''

x = torch.randn(1024, 512)
y = torch.randn(512, 64)

loop = 10000
start_time = datetime.now()

for i in list(range(loop)):
    z = torch.mm(x, y)

end_time = datetime.now()

start_ctime = datetime.ctime(start_time)
end_ctime = datetime.ctime(end_time)

print("%s Time cost:" % ("GPU" if torch.cuda.is_available() else "CPU"))
print("execute time: %s ~ %s" % (start_ctime, end_ctime))
print((end_time - start_time).total_seconds())

x = torch.randn(1024, 512).to(torch.device("cpu"))
y = torch.randn(512, 64).to(torch.device("cpu"))

start_time = datetime.now()

for i in list(range(loop)):
    z = torch.mm(x, y)

end_time = datetime.now()

start_ctime = datetime.ctime(start_time)
end_ctime = datetime.ctime(end_time)

print("CPU Time cost:")
print("execute time: %s ~ %s" % (start_ctime, end_ctime))
print((end_time - start_time).total_seconds())
