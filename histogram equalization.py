import cv2
import numpy as np

img = cv2.imread('color_img.jpg')

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_h, img_w, _ = img.shape

img_h, img_w = img_gray.shape

img_b, img_g, img_r = cv2.split(img)
img_b_hsv, img_g_hsv, img_r_hsv = cv2.split(img_hsv)

def perform_hist(img):
    histogram = np.zeros(256)
    for i in range(img_h):
        for j in range(img_w):
            intensity = img[i, j]
            histogram[intensity] += 1
    return histogram

def calculate_pdf(histogram):
    pdf = histogram / (img_h * img_w)
    return pdf

def calculate_cdf(pdf):
    cdf = np.cumsum(pdf) * 255
    return cdf

def histogram_equalization(img, cdf):
    output = np.zeros_like(img)
    for i in range(img_h):
        for j in range(img_w):
            output[i, j] = cdf[img[i, j]]
    return output


hist = perform_hist(img_gray)

hist_b = perform_hist(img_b)
hist_g = perform_hist(img_g)
hist_r = perform_hist(img_r)

hist_b_hsv = perform_hist(img_b_hsv)
hist_g_hsv = perform_hist(img_g_hsv)
hist_r_hsv = perform_hist(img_r_hsv)

pdf_gray = calculate_pdf(hist)

pdf_b = calculate_pdf(hist_b)
pdf_g = calculate_pdf(hist_g)
pdf_r = calculate_pdf(hist_r)
pdf_b_hsv = calculate_pdf(hist_b_hsv)
pdf_g_hsv = calculate_pdf(hist_g_hsv)
pdf_r_hsv = calculate_pdf(hist_r_hsv)



cdf_gray = calculate_cdf(pdf_gray)

cdf_b = calculate_cdf(pdf_b)
cdf_g = calculate_cdf(pdf_g)
cdf_r = calculate_cdf(pdf_r)
cdf_b_hsv = calculate_cdf(pdf_b_hsv)
cdf_g_hsv = calculate_cdf(pdf_g_hsv)
cdf_r_hsv = calculate_cdf(pdf_r_hsv)

equalized_img_gray = histogram_equalization(img_gray, cdf_gray)
equalized_img_gray = np.round(equalized_img_gray).astype(np.uint8)

equalized_img_b = histogram_equalization(img_b, cdf_b)
equalized_img_g = histogram_equalization(img_g, cdf_g)
equalized_img_r = histogram_equalization(img_r, cdf_r)

equalized_img_v_hsv = histogram_equalization(img_r_hsv, cdf_r_hsv)


equalized_img = cv2.merge([equalized_img_b, equalized_img_g, equalized_img_r])
equalized_img_hsv = cv2.merge([img_b_hsv, img_g_hsv, equalized_img_v_hsv])

equalized_img_hsv_to_color = cv2.cvtColor(equalized_img_hsv, cv2.COLOR_HSV2BGR)

cv2.imshow("Merged image", equalized_img)
cv2.imshow("Merged image HSV To Color", equalized_img_hsv_to_color)
cv2.imshow("Merged image Gray", equalized_img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
