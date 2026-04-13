"""
https://towardsdatascience.com/implementing-linear-regression-with-gradient-descent-from-scratch-f6d088ec1219/
https://www.geeksforgeeks.org/machine-learning/gradient-descent-in-linear-regression/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path =  'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.head(10)

print(type(data))
data.describe()

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
plt.show()

def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


print(type(data))
data.insert(0, 'Ones', 1)


# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]#X是所有行，去掉最后一列
y = data.iloc[:,cols-1:cols]#X是所有行，最后一列


X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))

print(theta.shape)


'''
获得参数个数
'''
# https://numpy.org/doc/2.2/reference/generated/numpy.ravel.html
# Return a contiguous flattened array.
# A 1-D array, containing the elements of the input, is returned. A copy is made only if needed.
parameters = int(theta.ravel().shape[1])

print(parameters)

print(theta) # [[0 0]]
print(theta.ravel()) # [[0 0]]

'''
超参数
'''
alpha = 0.01
iters = 10000

cost = np.zeros(iters)
# 每次迭代的损失，用 Vector 存储
print(cost)

# 计算损失
error = (X * theta.T) - y

'''
Numpy 中，做乘法运算的多种形式
https://zhuanlan.zhihu.com/p/480960607
'''
term = np.multiply(error, X[:,j])

