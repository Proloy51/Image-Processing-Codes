import numpy as np
import cv2
from matplotlib import pyplot as plt

def manual_convolution(image, kernel, center_x, center_y):
    if len(image.shape) == 2:  # Grayscale image
        return apply_kernel(image, kernel, center_x, center_y)
    elif len(image.shape) == 3:  # RGB image
        channels = cv2.split(image)
        conv_channels = [apply_kernel(channel, kernel, center_x, center_y) for channel in channels]
        return cv2.merge(conv_channels)

def apply_kernel(channel, kernel, center_x, center_y):
    kernel_height, kernel_width = kernel.shape

    pad_height_top = center_y
    pad_height_bottom = kernel_height - center_y - 1
    pad_width_left = center_x
    pad_width_right = kernel_width - center_x - 1

    padded_image = np.pad(channel, ((pad_height_top, pad_height_bottom), (pad_width_left, pad_width_right)), mode='constant', constant_values=0)

    output = np.zeros(channel.shape, dtype=np.float32)

    for i in range(channel.shape[0]):
        for j in range(channel.shape[1]):
            sum = 0
            for m in range(kernel_height):
                for n in range(kernel_width):
                    sum += kernel[m, n] * padded_image[i + m, j + n]

            output[i, j] = sum

    cv2.normalize(output, output, 0, 255, cv2.NORM_MINMAX)
    output = np.uint8(output)

    return output

kernel1 = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32)

kernel2 = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ], dtype=np.float32)

# Define the kernel center (e.g., center of a 3x3 kernel is at (1, 1))
center_x = 1
center_y = 1

image = cv2.imread('Lena.jpg')

image2 = manual_convolution(image, kernel1, center_x, center_y)
image3 = manual_convolution(image, kernel2, center_x, center_y)

cv2.imshow('1', image2)
cv2.waitKey(0)
cv2.imshow('2', image3)
cv2.waitKey(0)

cv2.imwrite('lena_gx.jpg', image2)
cv2.imwrite('lena_gy.jpg', image3)