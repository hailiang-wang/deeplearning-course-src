'''
如何求解两个矩阵的相似度？Hadamard product 然后求和
https://zhuanlan.zhihu.com/p/2059951288935244051
'''
import torch

a = torch.tensor([1.0, 2.0, 3.0]).float()
b = torch.tensor([4.0, 5.0, 6.0]).float()

###################
# 向量运算
###################
'''
Hadamard Product of vectors
# 也称为：Element-wise Product
'''

c = a * b
# 也可以使用 c = torch.mul(a,b) 或 c = torch.multiply(a,b)
print(c)
# tensor([ 4., 10., 18.])

'''
dot product of vectors
* inner product/内积包括很多中，用来将两个向量计算生成一个标量
* dot product of vectors 是内积运算的一种，见 https://dilipkumar.medium.com/linear-algebra-dot-and-inner-product-5989c3bf824c
'''
d = torch.dot(a, b)
print(d)
# tensor(32.)

if d.item() == torch.sum(c).item():
    print("d and c has equal values.")


###################
# 矩阵运算
###################
A = torch.tensor([[2, 4], [1, 3], [5, 2]]).float()
B = torch.tensor([[3, 1], [2, 4], [1, 6]]).float()
print("A", A)
# A tensor([[2., 4.],
#         [1., 3.],
#         [5., 2.]])

print("B", B)
# B tensor([[3., 1.],
#         [2., 4.],
#         [1., 6.]])

'''
Hadamard Product of matrices
# 也称为：Element-wise Product
'''
C = A * B
print("C", C)
# C tensor([[ 6.,  4.],
#         [ 2., 12.],
#         [ 5., 12.]])

'''
dot product of matrices
* 矩阵间的内积，就是矩阵乘法
* https://builtin.com/data-science/dot-product-matrix
'''
try:
    # torch.dot 只支持两个向量运算
    D = torch.dot(A, B)
    print("D", D)
    # RuntimeError: 1D tensors expected, but got 2D and 2D tensors
except BaseException as e:
    print("WARN -")
    print("  ", e)

# the dot product sums the element-wise products to produce a scalar (in the case of vectors)
# or applies matrix multiplication rules for matrices.
E = B.transpose(1, 0)

# 此时，A 和 E 满足乘法运算律，即 A 的列等于 E 的行
D = torch.mm(A, E)
print("D", D)


'''
Sum Hadamard Product of matrices
# 两个矩阵，按元素位置乘积，最后求和
* 用来作两个矩阵的相似度计算
  * 要先进行单位标准差/Smoothing
  * 可以进一步的在求和阶段，使用权值进行优化？
'''
e = torch.sum(A * B)
print("e", e)
