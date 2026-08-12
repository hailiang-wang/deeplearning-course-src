'''
掌握 Tensor 的乘法
'''
import torch

# 打印的样式
torch.set_printoptions(edgeitems=10, threshold=50)
torch.manual_seed(100)

'''
内积 / Inner product：两个向量之间进行
'''
x = torch.randn(2)  # 得到了一个行向量
print(x.shape)

y = x.unsqueeze(dim=0).transpose(dim0=1, dim1=0)
print(y.shape)

z = torch.randn(2)
print("x", x)
print("z", z)

b = torch.dot(x, z)
print("b", b)


'''
Hadamard Product / 阿达玛乘积
* 对应位置上相乘
'''
print("*" * 80)
print("Hadamard Product / 阿达玛乘积")
x = torch.randn(3, 2)
y = torch.randn(3, 2)
print("x", x)
print("y", y)

c = torch.multiply(x, y)
print("c", c)

p = x * y
print("p", p)

q = torch.mul(x, y)
print("q", q)

'''
阿达玛乘积 和 内积，在深度学习中，扮演非常重要的角色。
那么，对应的应用场景，是什么？
https://www.zhihu.com/question/584712501/answer/2041814597829670549
'''

##########################
# Matrix Product  矩阵乘法
##########################
print("*" * 80)
print("矩阵乘法")
x = torch.randn(3, 2)
y = torch.randn(2, 2)
print("x", x)
print("y", y)

z = torch.mm(x, y)  # matrix multiply
print(z)
print(z.shape)

# 矩阵乘法接口的限制：
# * 不支持广播
# * 仅仅能在两个矩阵之间运算，不支持 dim > 2 的情况

# 第二种，矩阵乘法的接口，就是改变了上面的限制
x = torch.randn(2, 3, 2)
y = torch.randn(2, 2, 2)
print("x", x)
print("y", y)

z = torch.matmul(x, y)
print("z", z)
print("z.shape", z.shape)
