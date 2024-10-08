import numpy as np
import cv2
import math


img = cv2.imread("C:/Users/prolo/OneDrive/Desktop/4-1/LAB/Image Processing Lab/Project/lena.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_kernal(sigma, kernel_size):
    h = 1/(2*math.pi*(sigma**2))
    n = kernel_size // 2 
    kernel = np.zeros((kernel_size,kernel_size))
    for i in range(-n,n+1):
        for j in range(-n,n+1):
            dist = (i**2 + j**2)/(2*math.pi*(sigma**2))
            kernel[i+n,j+n] = h*np.exp(-dist)
    return kernel
            

'''def convolution(img, kernel,p,q):
    n = kernel.shape[0] // 2 
    top = img.shape[0]-p-1
    left = img.shape[1]-q-1
    bottom=p
    right=q
    
    img_bordered = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)
    
    output = np.zeros((img.shape[0],img.shape[1]))
    
    for x in range(n, img.shape[0]-n):
        for y in range(n, img.shape[1]-n):
            sum = 0
            for i in range(-n,n+1):
                for j in range(-n, n+1):
                    sum += kernel[i+n,j+n] * img_bordered[x-i,y-j]
            output[x,y] = sum'''
            
            
def convolution(img, kernel,p,q):
    n = kernel.shape[0] // 2 
    top = img.shape[0]-p-1
    left = img.shape[1]-q-1
    bottom=p
    right=q
    
    img_bordered = cv2.copyMakeBorder(img, top=n, bottom=n, left=n, right=n, borderType=cv2.BORDER_CONSTANT)

    out = np.zeros((img_bordered.shape[0], img_bordered.shape[1]))

    for x in range(n, img_bordered.shape[0] - n):
        for y in range(n, img_bordered.shape[1] - n):
            sum = 0
            for i in range(-n, n + 1):
                for j in range(-n, n + 1):
                    if 0 <= x - i < img_bordered.shape[0] and 0 <= y - j < img_bordered.shape[1]:
                        sum += img_bordered[x - i, y - j] * kernel[i + n, j + n]
            out[x, y] = np.clip(sum,0,255)

    return out
            


kernel_size=5
sigma = 1
p = int(input("Enter x of the center : "))
q = int(input("ENter y of the center : "))

kernel = gaussian_kernal(sigma, kernel_size)
print(kernel)

output = convolution(img, kernel, 2, 2)
cv2.normalize(output, output,0,255,cv2.NORM_MINMAX)
output = np.round(output).astype(np.uint8)

cv2.imshow("Input image", img)
cv2.waitKey(0)
cv2.imshow("Output image", output)
cv2.waitKey(0)

cv2.destroyAllWindows()