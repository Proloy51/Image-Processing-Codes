import numpy as np
import cv2
import math

img_gray = cv2.imread("final_image.jpg",cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread("final_image.jpg", cv2.IMREAD_COLOR)
img_hsv = cv2.cvtColor(img_color, cv2.COLOR_RGB2HSV)
img_hsv_to_color = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

'''cv2.imshow("Hsv image", img_hsv)
cv2.imshow("Hsv to Color", img_hsv_to_color)
cv2.waitKey(0)'''

# kernel of Gaussian Filter
def gaussian_filter(sigma_x,sigma_y,kernel_size):
    """returns a gaussian blur filter"""

    kernel = np.zeros((kernel_size,kernel_size))

    h = 1/(2.0*math.pi*sigma_x*sigma_y)

    for i in range(kernel_size):
        for j in range(kernel_size):
            p = ((i**2)/(sigma_x**2)) + ((j**2)/(sigma_y**2))
            kernel[i,j] = h*np.exp(-0.5*p)

    print("Kernel of Gaussian Blur Filter : ")
    print(kernel)
    return kernel

# kernel of Mean filter

def mean_filter(kernel_size):
    kernel = (np.ones((kernel_size,kernel_size)))
    return (1/(kernel_size*kernel_size))*kernel


# kernel of Laplacian Filter

def laplacian_kernel(kernel_size,sign):
    """Returns the Laplacian kernel."""
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            if i == center and j == center:
                if sign == '-':
                    kernel[i, j] = -(kernel_size*kernel_size)
                if sign =='+':
                    kernel[i, j] = +(kernel_size*kernel_size)
            else:
                if sign=='-':
                    kernel[i, j] = 1
                if sign=='+':
                    kernel[i, j] = -1
    return kernel


# kernel of LoG filter



def laplacian_of_gaussian(sigma, kernel_size):

    n = kernel_size // 2

    kernel = np.zeros((kernel_size, kernel_size))

    h = -1 / (math.pi * (sigma**4))

    for i in range(-n, n+1):
        for j in range(-n, n+1):
            distance_squared = (i ** 2) + (j ** 2)

            kernel[i+n, j+n] = h * (1 - (distance_squared / (2 * (sigma ** 2)))) * math.exp(-distance_squared / (2 * (sigma ** 2)))

    return kernel



# kernel of Sobel filter

def sobel_kernel(size):
    """returns a horizontal sobel kernel and a vertical sobel kernel"""
    if size % 2 == 0:
        raise ValueError("Size must be an odd number.")

    sobel_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])

    sobel_y = np.array([
        [1, 2, 1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ])

    return sobel_x, sobel_y


#GrayScale image smoothing
def conv_smooth_gray(img_gray,kernel,p,q):
    top = kernel.shape[0]-p-1
    left = kernel.shape[1]-q-1
    bottom = p
    right = q

    img_gray_bordered = cv2.copyMakeBorder(img_gray, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT)

    output_smooth_gray =  np.zeros_like(img_gray_bordered)

   # n = kernel.shape[0] // 2

    h = img_gray.shape[0]
    w = img_gray.shape[1]
    kh = kernel.shape[0]
    kw = kernel.shape[1]


    for x in range (h):
        for y in range (w):
            out_val=0
            for i in range (kh):
                for j in range (kw):
                    #output_smooth_gray[x][y] += img_gray_bordered[x + i][y + j] * kernel[kh - 1 - i][kw - 1 - j]
                    out_val += img_gray_bordered[x + i][y + j] * kernel[kh - 1 - i][kw - 1 - j]

            output_smooth_gray[x][y] = np.clip(out_val, 0, 255)

    cv2.normalize(output_smooth_gray, output_smooth_gray, 0, 255, cv2.NORM_MINMAX)
    output_smooth_gray = np.round(output_smooth_gray).astype(np.uint8)
    return output_smooth_gray



# Color image smoothing
def conv_smooth_color(img_color,kernel,p,q):
    top = kernel.shape[0]-p-1
    left = kernel.shape[1]-q-1
    bottom = p
    right = q

    img_color_bordered = cv2.copyMakeBorder(img_color, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT)

    output_smooth_color =  np.zeros_like(img_color_bordered)

   # n = kernel.shape[0] // 2

    h = img_color.shape[0]
    w = img_color.shape[1]
    c = img_color.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]


    for ch in range(c):

        for x in range(h):
            for y in range(w):

                out_val = 0
                for i in range(kernel.shape[0]):
                    for j in range(kernel.shape[1]):

                        x_padded = x + i
                        y_padded = y + j

                        out_val += img_color_bordered[x_padded][y_padded][ch] * kernel[kh-1-i][kw-1-j]

                output_smooth_color[x][y][ch] = np.clip(out_val, 0, 255)

    cv2.normalize(output_smooth_color, output_smooth_color, 0, 255, cv2.NORM_MINMAX)
    output_smooth_color = np.round(output_smooth_color).astype(np.uint8)
    return output_smooth_color


# HSV image smoothing
def conv_smooth_hsv(img_hsv,kernel,p,q):
    top = kernel.shape[0]-p-1
    left = kernel.shape[1]-q-1
    bottom = p
    right = q

    img_hsv_bordered = cv2.copyMakeBorder(img_hsv, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT)

    output_smooth_hsv =  np.zeros_like(img_hsv_bordered)

    #n = kernel.shape[0] // 2

    h = img_hsv.shape[0]
    w = img_hsv.shape[1]
    c = img_hsv.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]


    for ch in range(c):

        for x in range(h):
            for y in range(w):

                out_val = 0
                for i in range(kernel.shape[0]):
                    for j in range(kernel.shape[1]):

                        x_padded = x + i
                        y_padded = y + j

                        out_val += img_hsv_bordered[x_padded][y_padded][ch] * kernel[kh-1-i][kw-1-j]

                output_smooth_hsv[x][y][ch] = np.clip(out_val, 0, 255)

    cv2.normalize(output_smooth_hsv, output_smooth_hsv, 0, 255, cv2.NORM_MINMAX)
    output_smooth_hsv = np.round(output_smooth_hsv).astype(np.uint8)
    return output_smooth_hsv

# Grayscale image sharpening

def conv_sharp_gray(img_gray,kernel,p,q):
    top = kernel.shape[0]-p-1
    left = kernel.shape[1]-q-1
    bottom = p
    right = q

    img_gray_bordered = cv2.copyMakeBorder(img_gray, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT)

    output_sharp_gray =  np.zeros_like(img_gray_bordered)

    n = kernel.shape[0] // 2

    for x in range(n, img_gray_bordered.shape[0]-n):
        for y in range(n, img_gray_bordered.shape[1]-n):
            res=0
            for i in range(-n, n+1):
                for j in range(-n, n+1):
                    ff = kernel[i+n, j+n]
                    ii = img_gray_bordered[x-i, y-j]
                    res += (ff*ii)
            output_sharp_gray[x,y] = np.clip(res, 0, 255)

    cv2.normalize(output_sharp_gray, output_sharp_gray, 0, 255, cv2.NORM_MINMAX)
    output_sharp_gray = np.round(output_sharp_gray).astype(np.uint8)
    return output_sharp_gray


# Color image sharpening

def conv_sharp_color(img_color,kernel,p,q):
    top = kernel.shape[0]-p-1
    left = kernel.shape[1]-q-1
    bottom = p
    right = q

    img_color_bordered = cv2.copyMakeBorder(img_color, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT)

    output_sharp_color =  np.zeros_like(img_color_bordered)


    n = kernel.shape[0] // 2

    for c in range(img_color_bordered.shape[2]):
        for x in range(n,img_color_bordered.shape[0]-n):
            for y in range(n,img_color_bordered.shape[1]-n):
                res=0
                for i in range(-n, n+1):
                    for j in range(-n, n+1):
                        ff = kernel[i+n, j+n]
                        ii = img_color_bordered[x-i,y-j,c]
                        res += (ff*ii)
                output_sharp_color[x,y,c] = np.clip(res, 0, 255)

    return output_sharp_color

def conv_sharp_color2(img_color,kernel,p,q):
    top = kernel.shape[0]-p-1
    left = kernel.shape[1]-q-1
    bottom = p
    right = q

    img_color_bordered = cv2.copyMakeBorder(img_color, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT)

    output_sharp_color =  np.zeros_like(img_color_bordered)


    output_sharp_color = cv2.filter2D(src=img_color, ddepth=-1, kernel=kernel)

    cv2.normalize(output_sharp_color, output_sharp_color, 0, 255, cv2.NORM_MINMAX)
    output_sharp_color = np.round(output_sharp_color).astype(np.uint8)
    return output_sharp_color


def conv_smooth_color2(img_color,kernel,p,q):
    top = kernel.shape[0]-p-1
    left = kernel.shape[1]-q-1
    bottom = p
    right = q

    img_color_bordered = cv2.copyMakeBorder(img_color, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT)

    output_smooth_color =  np.zeros_like(img_color_bordered)

    output_smooth_color = cv2.filter2D(src=img_color, ddepth=-1, kernel=kernel)

    cv2.normalize(output_smooth_color, output_smooth_color, 0, 255, cv2.NORM_MINMAX)
    output_smooth_color = np.round(output_smooth_color).astype(np.uint8)
    return output_smooth_color



while(1):
    print("1. Color Image")
    print("2. GrayScale Image")
    print("3. HSV Image")
    print("0. Back")

    i = int(input())
    test = i


    if i==0:
        break
    if i==1:
        print("1. Smooth")
        print("2. Sharp")
        print("0. Back")

        j = int(input())

        if j==0:
            break

        if j==1:
            while(1):
                print("1. Gaussian Blur filter")
                print("2. Mean Filter")
                print("0. Back")

                k = int(input())

                if k==0:
                    break
                if k==1:
                    #print("Enter sigma and kernel Size : ")
                    print("Enter sigma_x, sigma_y and kernel Size : ")
                    sigma_x = float(input())
                    sigma_y = float(input())
                    kernel_size = int(input())

                    kernel = gaussian_filter(sigma_x, sigma_y, kernel_size)

                    print("Enter the center of the kernel : ")

                    p = int(input())
                    q = int(input())

                    output_smooth_color = conv_smooth_color(img_color, kernel,p,q)
                    output_smooth_hsv = conv_smooth_color(img_hsv, kernel,p,q)
                    #output_smooth_hsv = conv_smooth_hsv(img_hsv, kernel, p, q)
                    output_smooth_hsv_to_color = cv2.cvtColor(output_smooth_hsv, cv2.COLOR_HSV2RGB)

                    cv2.imshow("Input Color Image", img_color)
                    cv2.imshow("Color Smooth Gaussian Filter Image", output_smooth_color)
                    cv2.waitKey(0)


                    print("1. To see the Difference between color and HSV convolution")
                    print("0. Back")

                    p = int(input())

                    if p==1:
                        diff_gaussian = np.abs(output_smooth_color - output_smooth_hsv_to_color)
                        cv2.imshow("Difference in Gaussian Filter", diff_gaussian)
                        cv2.waitKey(0)
                    if p==0:
                        break

                if k==2:
                    print("Enter the kernel size : ")
                    kernel_size = int(input())
                    kernel = mean_filter(kernel_size)
                    print("Kernel of Mean filter : ")
                    print(kernel)

                    print("Enter the center of the kernel : ")

                    p = int(input())
                    q = int(input())

                    output_smooth_color = conv_smooth_color(img_color, kernel,p,q)
                    output_smooth_hsv = conv_smooth_color(img_hsv, kernel,p,q)
                    output_smooth_hsv_to_color = cv2.cvtColor(output_smooth_hsv, cv2.COLOR_HSV2RGB)

                    cv2.imshow("Input Color Image", img_color)
                    cv2.imshow("Color Smooth Mean Filter Image", output_smooth_color)
                    cv2.waitKey(0)

                    print("1. To see the Difference between color and HSV convolution")
                    print("0. Back")

                    p = int(input())

                    if p==1:
                        diff_mean = (output_smooth_color - output_smooth_hsv_to_color)
                        cv2.imshow("Difference in Mean Filter", diff_mean)
                        cv2.waitKey(0)
                    if p==0:
                        break

        if j==2:
            print("1. Laplacian Filter")
            print("2. LoG Filter")
            print("3. Sobel Filter")
            print("0. Back")

            k = int(input())

            if k==0:
                break

            if k==1:
                print("Enter the kernel size : ")
                kernel_size = int(input())
                print("Enter the sign of the center coefficient : ")
                sign = input()
                kernel = laplacian_kernel(kernel_size,sign)
                print("Kernel of Laplacian Filter : ")
                print(kernel)

                print("Enter the center of the kernel : ")

                p = int(input())
                q = int(input())

                output_sharp_color = conv_sharp_color(img_color, kernel,p,q)
                output_sharp_hsv = conv_sharp_color(img_hsv, kernel,p,q)
                output_sharp_hsv_to_color = cv2.cvtColor(output_sharp_hsv, cv2.COLOR_HSV2RGB)
 
                cv2.imshow("Input Color Image", img_color)
                cv2.imshow("Color Sharp Laplacian Filter", output_sharp_color)
                cv2.waitKey(0)

                print("1. To see the Difference between color and HSV convolution")
                print("0. Back")

                p = int(input())

                if p==1:
                    diff_laplacian = np.abs(output_sharp_color - output_sharp_hsv_to_color)
                    cv2.imshow("Difference in Laplacian Filter", diff_laplacian)
                    cv2.waitKey(0)
                if p==0:
                    break

            if k==2:
                #print("Enter sigma,kernel Size : ")
                print("Enter sigma_x, sigma_y and kernel size : ")

                #sigma = float(input())
                sigma_x = float(input())
                sigma_y = float(input())
                kernel_size = int(input())

                kernel = laplacian_of_gaussian(sigma_x, kernel_size)
               # kernel = laplacian_of_gaussian(sigma_x, sigma_y, kernel_size)
                print("Kernel of Laplacian of Gaussian(LoG) Filter : ")
                print(kernel)

                print("Enter the center of the kernel : ")

                p = int(input())
                q = int(input())

                output_sharp_color = conv_sharp_color(img_color, kernel,p,q)
                output_sharp_hsv = conv_sharp_color(img_hsv, kernel,p,q)
                output_sharp_hsv_to_color = cv2.cvtColor(output_sharp_hsv, cv2.COLOR_HSV2RGB)

                cv2.imshow("Input Color Image", img_color)
                cv2.imshow("Color Sharp LoG Filter", output_sharp_color)
                cv2.waitKey(0)

                print("1. To see the Difference between color and HSV convolution")
                print("0. Back")

                p = int(input())

                if p==1:
                    diff_laplacian = np.abs(output_sharp_color - output_sharp_hsv_to_color)
                    cv2.imshow("Difference in LoG Filter", diff_laplacian)
                    cv2.waitKey(0)
                if p==0:
                    break

            if k==3:
                print("Enter the kernel size : ")
                kernel_size = int(input())

                sobel_x, sobel_y = sobel_kernel(kernel_size)

                print("Enter the center of the kernel : ")

                p = int(input())
                q = int(input())

                output_sharp_color_x = conv_sharp_color(img_color, sobel_x,p,q)
                output_sharp_color_y = conv_sharp_color(img_color, sobel_y,p,q)
                output_sharp_hsv_x = conv_sharp_color(img_hsv, sobel_x,p,q)
                output_sharp_hsv_to_color_x = cv2.cvtColor(output_sharp_hsv_x, cv2.COLOR_HSV2RGB)
                output_sharp_hsv_y = conv_sharp_color(img_hsv, sobel_y,p,q)
                output_sharp_hsv_to_color_y = cv2.cvtColor(output_sharp_hsv_y, cv2.COLOR_HSV2RGB)

                print("1. Horizontal")
                print("2. Vertical")

                q = int(input())
                if q==1:
                    #cv2.imshow("Input Color Image", img_color)
                    #cv2.imshow("Input HSV Image", img_hsv)
                    cv2.imshow("Color Sharp Sobel_X Filter", output_sharp_color_x)
                    cv2.imshow("HSV Sharp Sobel_X Filter", output_sharp_hsv_to_color_x)
                    diff_sobel_x = np.abs(output_sharp_color_x - output_sharp_hsv_to_color_x)

                    cv2.normalize(diff_sobel_x, diff_sobel_x, 0, 255, cv2.NORM_MINMAX)
                    diff_sobel_x = np.round(diff_sobel_x).astype(np.uint8)

                    cv2.imshow("Difference in Sobel Filter", diff_sobel_x)
                    cv2.waitKey(0)

                    print("1. To see the Difference between color and HSV convolution")
                    print("0. Back")

                    p = int(input())

                    if p==1:
                        diff_sobel_x = np.abs(output_sharp_color_x - output_sharp_hsv_to_color_x)


                        cv2.imshow("Difference in Sobel Filter", diff_sobel_x)
                        cv2.waitKey(0)
                    if p==0:
                        break


                if q==2:
                    cv2.imshow("Input Color Image", img_color)
                    cv2.imshow("Color Sharp Sobel_Y Filter", output_sharp_color_y)
                    cv2.waitKey(0)

                    print("1. To see the Difference between color and HSV convolution")
                    print("0. Back")

                    p = int(input())

                    if p==1:
                        diff_sobel_y = np.abs(output_sharp_color_y - output_sharp_hsv_to_color_y)
                        cv2.imshow("Difference in Laplacian Filter", diff_sobel_y)
                        cv2.waitKey(0)
                    if p==0:
                        break




    if i==3:
        print("1. Smooth")
        print("2. Sharp")
        print("0. Back")

        j = int(input())

        if j==0:
            break

        if j==1:
            while(1):
                print("1. Gaussian Blur filter")
                print("2. Mean Filter")
                print("0. Back")

                k = int(input())

                if k==0:
                    break
                if k==1:
                    #print("Enter sigma and kernel Size : ")
                    print("Enter sigma_x, sigma_y and kernel Size : ")
                    sigma_x = float(input())
                    sigma_y = float(input())
                    kernel_size = int(input())

                    kernel = gaussian_filter(sigma_x,sigma_y, kernel_size)

                    print("Enter the center of the kernel : ")

                    p = int(input())
                    q = int(input())

                    output_smooth_color = conv_smooth_color(img_color, kernel,p,q)
                    output_smooth_hsv = conv_smooth_color(img_hsv, kernel,p,q)
                    output_smooth_hsv_to_color = cv2.cvtColor(output_smooth_hsv, cv2.COLOR_HSV2RGB)

                    cv2.imshow("Input HSV Image", img_hsv_to_color)
                    cv2.imshow("HSV Smooth Gaussian Filter Image", output_smooth_hsv_to_color)
                    cv2.waitKey(0)

                    print("1. To see the Difference between color and HSV convolution")
                    print("0. Back")

                    p = int(input())

                    if p==1:
                        diff_gaussian = np.abs(output_smooth_color - output_smooth_hsv_to_color)
                        cv2.imshow("Difference in Gaussian Filter", diff_gaussian)
                        cv2.waitKey(0)
                    if p==0:
                        break

                if k==2:
                    print("Enter the kernel size : ")
                    kernel_size = int(input())
                    kernel = mean_filter(kernel_size)
                    print("Kernel of Mean filter : ")
                    print(kernel)

                    print("Enter the center of the kernel : ")

                    p = int(input())
                    q = int(input())

                    output_smooth_color = conv_smooth_color(img_color, kernel,p,q)
                    output_smooth_hsv = conv_smooth_color(img_hsv, kernel,p,q)
                    output_smooth_hsv_to_color = cv2.cvtColor(output_smooth_hsv, cv2.COLOR_HSV2RGB)

                    cv2.imshow("Input HSV Image", img_hsv)
                    cv2.imshow("HSV Smooth Mean Filter Image", output_smooth_hsv_to_color)
                    cv2.waitKey(0)

                    print("1. To see the Difference between color and HSV convolution")
                    print("0. Back")

                    p = int(input())

                    if p==1:
                        diff_mean = np.abs(output_smooth_color - output_smooth_hsv_to_color)
                        cv2.imshow("Difference in Mean Filter", diff_mean)
                        cv2.waitKey(0)
                    if p==0:
                        break

        if j==2:
            print("1. Laplacian Filter")
            print("2. LoG Filter")
            print("3. Sobel Filter")
            print("0. Back")

            k = int(input())

            if k==0:
                break

            if k==1:
                print("Enter the kernel size : ")
                kernel_size = int(input())
                print("Enter the sign of the center coefficient : ")
                sign = input()
                kernel = laplacian_kernel(kernel_size,sign)
                print("Kernel of Laplacian Filter : ")
                print(kernel)

                print("Enter the center of the kernel : ")

                p = int(input())
                q = int(input())

                output_sharp_color = conv_sharp_color(img_color, kernel,p,q)
                output_sharp_hsv = conv_sharp_color(img_hsv, kernel,p,q)
                output_sharp_hsv_to_color = cv2.cvtColor(output_sharp_hsv, cv2.COLOR_HSV2RGB)

                cv2.imshow("Input HSV Image", img_hsv)
                cv2.imshow("HSV Sharp Laplacian Filter Image", output_sharp_hsv_to_color)
                cv2.waitKey(0)

                print("1. To see the Difference between color and HSV convolution")
                print("0. Back")

                p = int(input())

                if p==1:
                    diff_laplacian = np.abs(output_sharp_color - output_sharp_hsv_to_color)
                    cv2.imshow("Difference in Laplacian Filter", diff_laplacian)
                    cv2.waitKey(0)
                if p==0:
                    break

            if k==2:
                #print("Enter sigma,kernel Size : ")
                print("Enter sigma_x, sigma_y and kernel size : ")

                #sigma = float(input())
                sigma_x = float(input())
                sigma_y = float(input())
                kernel_size = int(input())

                #kernel = laplacian_of_gaussian(sigma, kernel_size)
                kernel = laplacian_of_gaussian(sigma_x, kernel_size)
                print("Kernel of Laplacian of Gaussian(LoG) Filter : ")
                print(kernel)

                print("Enter the center of the kernel : ")

                p = int(input())
                q = int(input())

                output_sharp_color = conv_sharp_color(img_color, kernel,p,q)
                output_sharp_hsv = conv_sharp_color(img_hsv, kernel,p,q)
                output_sharp_hsv_to_color = cv2.cvtColor(output_sharp_hsv, cv2.COLOR_HSV2RGB)

                cv2.imshow("Input HSV Image", img_hsv)
                cv2.imshow("HSV Sharp LoG Filter Image", output_sharp_hsv_to_color)
                #cv2.imshow("HSV Sharp LoG Filter Image", output_sharp_hsv)
                cv2.waitKey(0)

                print("1. To see the Difference between color and HSV convolution")
                print("0. Back")

                p = int(input())

                if p==1:
                    diff_laplacian = np.abs(output_sharp_color - output_sharp_hsv_to_color)
                    cv2.imshow("Difference in Log Filter", diff_laplacian)
                    cv2.waitKey(0)
                if p==0:
                    break

            if k==3:
                print("Enter the kernel size : ")
                kernel_size = int(input())

                sobel_x, sobel_y = sobel_kernel(kernel_size)

                print("Enter the center of the kernel : ")

                p = int(input())
                q = int(input())

                output_sharp_color_x = conv_sharp_color(img_color, sobel_x,p,q)
                output_sharp_color_y = conv_sharp_color(img_color, sobel_y,p,q)
                output_sharp_hsv_x = conv_sharp_color(img_hsv, sobel_x,p,q)
                output_sharp_hsv_to_color_x = cv2.cvtColor(output_sharp_hsv_x, cv2.COLOR_HSV2RGB)
                output_sharp_hsv_y = conv_sharp_color(img_hsv, sobel_y,p,q)
                output_sharp_hsv_to_color_y = cv2.cvtColor(output_sharp_hsv_y, cv2.COLOR_HSV2RGB)
                print("1. Horizontal")
                print("2. Vertical")

                q = int(input())
                if q==1:
                    cv2.imshow("Input HSV Image", img_hsv)
                    cv2.imshow("HSV Sharp Sobel_X Filter Image", output_sharp_hsv_to_color_x)
                    cv2.waitKey(0)

                    print("1. To see the Difference between color and HSV convolution")
                    print("0. Back")

                    p = int(input())

                    if p==1:
                        diff_sobel_x = np.abs(output_sharp_color_x - output_sharp_hsv_to_color_x)
                        cv2.imshow("Difference in Sobel Filter", diff_sobel_x)
                        cv2.waitKey(0)
                    if p==0:
                        break


                if q==2:
                    cv2.imshow("Input HSV Image", img_hsv)
                    cv2.imshow("HSV Sharp Sobel_Y Filter Image", output_sharp_hsv_to_color_y)
                    cv2.waitKey(0)

                    print("1. To see the Difference between color and HSV convolution")
                    print("0. Back")

                    p = int(input())

                    if p==1:
                        diff_sobel_y = np.abs(output_sharp_color_y - output_sharp_hsv_to_color_y)
                        cv2.imshow("Difference in Sobel Filter", diff_sobel_y)
                        cv2.waitKey(0)
                    if p==0:
                        break

    if i==2:
        while(1):
            print("1. Smooth")
            print("2. Sharp")
            print("0. Back")

            #os.system('cls')
            j = int(input())

            if j==0:
                break
            if j==1:
                while(1):
                    print("1. Gaussian Blur filter")
                    print("2. Mean Filter")
                    print("0. Back")

                    k = int(input())

                    if k==0:
                        break

                    if k==1:
                        #print("Enter sigma and kernel Size : ")
                        print("Enter sigma_x, sigma_y and kernel Size : ")
                        sigma_x = float(input())
                        sigma_y = float(input())
                        kernel_size = int(input())

                        kernel = gaussian_filter(sigma_x, sigma_y, kernel_size)
                        print("Enter the center of the kernel : ")

                        p = int(input())
                        q = int(input())

                        output_smooth_gray = conv_smooth_gray(img_gray, kernel,p,q)
                        cv2.imshow("Input Grayscale Image", img_gray)
                        cv2.imshow("GrayScale Smooth Gaussian Filter", output_smooth_gray)
                        cv2.waitKey(0)

                    if k==2:
                        print("Enter the kernel size : ")
                        kernel_size = int(input())
                        kernel = mean_filter(kernel_size)
                        print("Kernel of Mean filter : ")
                        print(kernel)

                        print("Enter the center of the kernel : ")

                        p = int(input())
                        q = int(input())

                        output_smooth_gray = conv_smooth_gray(img_gray, kernel,p,q)
                        cv2.imshow("Input Grayscale Image", img_gray)
                        cv2.imshow("GrayScale Smooth Mean Filter", output_smooth_gray)
                        cv2.waitKey(0)

            if j==2:
                print("1. Laplacian Filter")
                print("2. LoG Filter")
                print("3. Sobel Filter")
                print("0. Back")

                k = int(input())

                if k==0:
                    break

                if k==1:
                    print("Enter the kernel size : ")
                    kernel_size = int(input())
                    print("Enter the sign of the center coefficient : ")
                    sign = input()
                    kernel = laplacian_kernel(kernel_size,sign)
                    print("Kernel of Laplacian Filter : ")
                    print(kernel)

                    print("Enter the center of the kernel : ")

                    p = int(input())
                    q = int(input())
                    output_sharp_gray = conv_sharp_gray(img_gray, kernel,p,q)
                    cv2.imshow("Input Grayscale Image", img_gray)
                    cv2.imshow("GrayScale Sharp Laplacian Filter", output_sharp_gray)
                    cv2.waitKey(0)

                if k==2:
                    #print("Enter sigma,kernel Size : ")
                    print("Enter sigma_x, sigma_y and kernel size : ")

                    #sigma = float(input())
                    sigma_x = float(input())
                    sigma_y = float(input())
                    kernel_size = int(input())

                    #kernel = laplacian_of_gaussian(sigma, kernel_size)
                    kernel = laplacian_of_gaussian(sigma_x,  kernel_size)
                    print("Kernel of Laplacian of Gaussian(LoG) Filter : ")
                    print(kernel)

                    print("Enter the center of the kernel : ")

                    p = int(input())
                    q = int(input())
                    output_sharp_gray = conv_sharp_gray(img_gray, kernel,p,q)
                    cv2.imshow("Input Grayscale Image", img_gray)
                    cv2.imshow("GrayScale Sharp LoG Filter", output_sharp_gray)
                    cv2.imwrite("Log_on_grayscale.png", output_sharp_gray)
                    cv2.waitKey(0)

                if k==3:
                    print("Enter the kernel size : ")
                    kernel_size = int(input())

                    sobel_x, sobel_y = sobel_kernel(kernel_size)

                    print("Enter the center of the kernel : ")

                    p = int(input())
                    q = int(input())

                    output_sharp_gray_x = conv_sharp_gray(img_gray, sobel_x,p,q)
                    cv2.imwrite("sobel_x_ob_grayscale.png", output_sharp_gray_x)
                    output_sharp_gray_y = conv_sharp_gray(img_gray, sobel_y,p,q)

                    print("1. Horizontal")
                    print("2. Vertical")

                    q = int(input())
                    if q==1:
                         cv2.imshow("Input Grayscale Image", img_gray)
                         cv2.imshow("GrayScale Sharp Sobel_X Filter", output_sharp_gray_x)
                         cv2.waitKey(0)
                    if q==2:
                         cv2.imshow("Input Grayscale Image", img_gray)
                         cv2.imshow("GrayScale Sharp Sobel_Y Filter", output_sharp_gray_y)
                         cv2.waitKey(0)


cv2.destroyAllWindows()


