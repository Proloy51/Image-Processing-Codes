import math
import numpy as np
import cv2

def x_derivatives(sigma, kernel_size):
    kernel = np.zeros((kernel_size, kernel_size))
    n = kernel_size // 2
    h = 1 / (2.0 * math.pi * (sigma ** 2))

    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            p = ((i ** 2) + (j ** 2)) / (2 * (sigma ** 2))
            kernel[i + n, j + n] = (-i / (sigma ** 2)) * h * np.exp(-1.0 * p)
            
    return kernel

def y_derivatives(sigma, kernel_size):
    kernel = np.zeros((kernel_size, kernel_size))
    n = kernel_size // 2
    h = 1 / (2.0 * math.pi * (sigma ** 2))

    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            p = ((i ** 2) + (j ** 2)) / (2 * (sigma ** 2))
            kernel[i + n, j + n] = (-j / (sigma ** 2)) * h * np.exp(-1.0 * p)

    return kernel

def convolutionGray(img, kernel):
    n = kernel.shape[0] // 2
    img_bordered = cv2.copyMakeBorder(img, top=n, bottom=n, left=n, right=n, borderType=cv2.BORDER_CONSTANT)

    out = np.zeros((img_bordered.shape[0], img_bordered.shape[1], 1))

    for x in range(n, img_bordered.shape[0] - n):
        for y in range(n, img_bordered.shape[1] - n):
            sum = 0
            for i in range(-n, n + 1):
                for j in range(-n, n + 1):
                    if 0 <= x - i < img_bordered.shape[0] and 0 <= y - j < img_bordered.shape[1]:
                        sum += img_bordered[x - i, y - j] * kernel[i + n, j + n]
            out[x, y] = sum

    return out

img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
sigma = float(input("sigma = "))
kernel_size = int(input("kernel size = "))

x_kernel = x_derivatives(sigma, kernel_size)
y_kernel = y_derivatives(sigma, kernel_size)

x_derivative = convolutionGray(img, x_kernel)
y_derivative = convolutionGray(img, y_kernel)

magnitude = np.sqrt(x_derivative ** 2 + y_derivative ** 2)

cv2.normalize(x_derivative,x_derivative,0,255,cv2.NORM_MINMAX)
x_derivative = np.round(x_derivative).astype(np.uint8)

cv2.normalize(y_derivative,y_derivative,0,255,cv2.NORM_MINMAX)
y_derivative = np.round(y_derivative).astype(np.uint8)

cv2.normalize(magnitude,magnitude,0,255,cv2.NORM_MINMAX)
magnitude = np.round(magnitude).astype(np.uint8)

# Apply global thresholding to the gradient magnitude image
threshold_value = 100  # You can adjust this value as needed
_, thresholded_magnitude = cv2.threshold(magnitude, threshold_value, 255, cv2.THRESH_BINARY)


cv2.imshow("x_derivative", x_derivative)
cv2.imshow("y_derivative", y_derivative)
cv2.imshow("gradient magnitude", magnitude.astype(np.uint8))
cv2.imshow("Thresholded Image",thresholded_magnitude)
cv2.waitKey(0)
