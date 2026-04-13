import numpy as np

'''
向量点积或矩阵乘法，shape不合法时报错
'''
m1 = np.matrix([[1, 2],
                [4, 5]])
m2 = np.matrix([[7],
                [8]])
# shape: m1(2x2) x m2(2x1) = 2x1
c = np.dot(m1, m2)  # 向量点积或矩阵乘法，shape不合法时报错
print(c)
