'''
实现一个卷积运算，对图片进行操作

Image processing 
https://en.wikipedia.org/wiki/Kernel_(image_processing)

Guide for convolution 2d:
https://medium.com/@whyamit404/numpy-2d-convolution-a-practical-guide-b01a759712e2

Guide for permute:
https://medium.com/@whyamit404/a-practical-guide-on-numpy-permute-22b72b4b7727

'''

import sys
from PIL import Image
import numpy as np

data_path = "data/p1ch8/original.png"
img_original = Image.open(data_path)
img = np.asarray(img_original)

img = np.transpose(img, (2, 0, 1))
print(img.shape)

'''
卷积核
'''
# 检测边缘
kernel1 = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1],
])

# 高斯模糊
kernel2 = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]) / 16


'''
卷积函数
'''


def conv2d(input_matrix, kernel):
    '''
    执行卷积
    '''
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    output_height = input_height + - kernel_height + 1
    output_width = input_width + - kernel_width + 1

    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            region = input_matrix[i:i + kernel_height, j:j + kernel_width]

            output[i, j] = max(min(np.sum(region * kernel), 255), 0)

    return output


def conv2d_with_padding(input_matrix, kernel, padding=1):
    '''
    对矩阵进行卷积运算，支持 padding
    '''
    matrix_padded = np.pad(input_matrix, pad_width=padding, mode="constant", constant_values=0)

    return conv2d(matrix_padded, kernel)


'''
测试卷积
'''


def process_imgs(img_array, kernels):
    '''
    针对图片 img，进行多次卷积运算
    '''
    for (k, out) in kernels:
        post_conv2d_img = np.zeros_like(img_array)

        for channel in range(img_array.shape[0]):
            post_conv2d_img[channel] = conv2d_with_padding(img_array[channel], k, padding=1)

        post_conv2d_img = np.transpose(post_conv2d_img, (1, 2, 0))
        # print("post_conv2d_img shape", post_conv2d_img.shape)

        post_img = Image.fromarray(post_conv2d_img)
        post_img.save(out)


process_imgs(img, [
    (kernel1, "data/p1ch8/post_kernel1.png"),
    (kernel2, "data/p1ch8/post_kernel2.png"),
])
