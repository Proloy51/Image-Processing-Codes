# Fourier transform - guassian lowpass filter

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

# take input
img_input = cv2.imread('two_noise.jpeg', 0)
img = img_input.copy()
image_size = img.shape[0] * img.shape[1]


#%%
# fourier transform

ft = np.fft.fft2(img)
ft_shift = np.fft.fftshift(ft)

magnitude_spectrum_ac = np.abs(ft_shift)
magnitude_spectrum = 20 * np.log(np.abs(ft_shift)+1)

ang = np.angle(ft_shift)
ang_ = cv2.normalize(ang, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U) 

final_result = np.multiply(magnitude_spectrum_ac, np.exp(1j*ang))

img_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
img_back_scaled = cv2.normalize(img_back, None, 0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)

x = 272
y = 256
r = 5

m,n = img.shape[0],img.shape[1]

filter=np.zeros((m,n),dtype=np.uint8)

for i in range(m):
    for j in range(n):
        distance = math.sqrt((i-x-m/2)**2+(j-y-n/2)**2)
        if(distance<=r):
            filter[i,j]=0
        else:
            filter[i,j]=1
  
# magnitude_spectrum = cv2.normalize(magnitude_spectrum, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U) 

notched = np.multiply(magnitude_spectrum,filter)


output = np.multiply(notched,np.exp(1j*ang))


output = np.real(np.fft.ifft2(np.fft.ifftshift(output)))

output = cv2.normalize(output,None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
# output = np.round(output).astype(np.uint8)

magnitude_spectrum = cv2.normalize(magnitude_spectrum, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U) 
cv2.imshow("input", img_input)
cv2.imshow("Magnitude Spectrum",magnitude_spectrum)
cv2.imshow("Phase", ang_)

cv2.imshow("notched",notched)
cv2.imshow("output",output)

cv2.waitKey(0)
cv2.destroyAllWindows()