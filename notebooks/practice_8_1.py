'''
Image processing 
https://en.wikipedia.org/wiki/Kernel_(image_processing)

Guide for convolution 2d:
https://medium.com/@whyamit404/numpy-2d-convolution-a-practical-guide-b01a759712e2

Guide for permute:
https://medium.com/@whyamit404/a-practical-guide-on-numpy-permute-22b72b4b7727

'''

from PIL import Image
import numpy as np

data_path = "data/p1ch8/original.png"

img = Image.open(data_path)
img_ndarray = np.asarray(img)

img_t = np.transpose(img_ndarray, (2, 1, 0))

# print(img_t.shape)

# Define the convolution function
def convolution2d(input_matrix, kernel):
    # Get dimensions of input and kernel
    input_h, input_w = input_matrix.shape
    kernel_h, kernel_w = kernel.shape

    # Calculate the size of the output matrix
    output_h = input_h - kernel_h + 1
    output_w = input_w - kernel_w + 1

    # Create an empty output matrix
    output = np.zeros((output_h, output_w))

    # Perform the convolution operation
    for i in range(output_h):
        for j in range(output_w):
            # Extract the region of the input matrix covered by the kernel
            region = input_matrix[i:i + kernel_h, j:j + kernel_w]

            # Apply element-wise multiplication and sum the result
            output[i, j] = np.sum(region * kernel)

    return output

# Add padding logic to the convolution function


def convolution2d_with_padding(input_matrix, kernel, padding=0):
    # Add padding to the input matrix
    input_padded = np.pad(input_matrix, pad_width=padding, mode='constant', constant_values=0)

    # Call the convolution function with the padded input
    return convolution2d(input_padded, kernel)


########################
# Define a 3x3 kernels (filter)
########################

# 边缘检测
kernel1 = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])

kernel2 = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

kernel3 = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]) / 16


def process_conv2d_with_kernels(kernels, output_paths):
    '''
    同时处理多个卷积核
    '''
    global img_t
    for kernel, output_path in zip(kernels, output_paths):
        post_img = np.zeros_like(img_t)

        for channel in range(img_t.shape[0]):
            post_img[channel] = convolution2d_with_padding(img_t[channel], kernel, 1)

        post_img_ndarray = np.transpose(post_img, (2, 1, 0))
        # print(post_img_ndarray.shape)

        transformed_image = Image.fromarray(post_img_ndarray)
        transformed_image.save(output_path)

process_conv2d_with_kernels([kernel1,
                             kernel2,
                             kernel3],
                            ["data/p1ch8/kernel1.png",
                             "data/p1ch8/kernel2.png",
                             "data/p1ch8/kernel3.png"])
