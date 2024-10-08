import numpy as np
import math
import cv2

def laplacian_of_gaussian(sigma, kernel_size):

    n = kernel_size // 2

    kernel = np.zeros((kernel_size, kernel_size))

    h = -1 / (math.pi * (sigma**4))

    for i in range(-n, n+1):
        for j in range(-n, n+1):
            distance_squared = (i ** 2) + (j ** 2)

            kernel[i+n, j+n] = h * (1 - (distance_squared / (2 * (sigma ** 2)))) * math.exp(-distance_squared / (2 * (sigma ** 2)))

    return kernel



def convolution(img, kernel):
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

# Example usage:

print("Enter value of Sigma : ")
sigma = int(input())

print("Enter size of the kernel : ")
kernel_size = int(input())

print("Enter threshold value : ")
threshold = int(input())

log_kernel = laplacian_of_gaussian(sigma, kernel_size)

img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
#cv2.imwrite("Lena_gray_input_image.png", img)
output = convolution(img, log_kernel)

output_an = convolution(img, log_kernel)

cv2.normalize(output_an, output_an, 0, 255, cv2.NORM_MINMAX)
output_an = np.round(output_an).astype(np.uint8)

def zero_crossing(image):
    h=image.shape[0]
    w = image.shape[1]
    zero_output = np.zeros_like(image)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            neighbors_x = [image[y - 1, x], image[y + 1, x]]
            neighbors_y = [image[y, x - 1], image[y, x + 1]]
            '''if any(val * image[y, x] < 0 for val in neighbors_x):
                zero_output[y, x] = image[y, x]
            elif any(val * image[y, x] < 0 for val in neighbors_y):
                zero_output[y, x] = image[y, x]'''
            if(neighbors_x[0]*neighbors_x[1] <0):
                zero_output[y,x] = image[y,x]
            elif(neighbors_y[0]*neighbors_y[1] <0):
                zero_output[y,x] = image[y,x]
    return zero_output

def simple_thresholding(image, kernel, threshold):
    n = kernel.shape[0] // 2
    h = image.shape[0]
    w = image.shape[1]
    thres_image = np.zeros((h,w))

    for x in range(1, h-1):
        for y in range(1, w-1):
            local_region = image[x-n:x+n+1, y-n:y+n+1]
            local_stddev = np.std(local_region)
            if local_stddev > threshold:
                thres_image[x, y] = image[x, y]

    return thres_image



zero_output = zero_crossing(output)
thres_output = simple_thresholding(zero_output, log_kernel, threshold)

cv2.imshow("zero_output", zero_output)
cv2.imshow("Log", output_an)
cv2.imshow("Threshold", thres_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
