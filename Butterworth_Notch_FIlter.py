# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 17:42:34 2024

@author: ASUS
"""

# Fourier transform - guassian lowpass filter

import cv2
import numpy as np
from math import sqrt

def point_op(img_size,uk,vk,D0,n):
    H=np.ones(img_size,dtype=np.float32)
    M=img_size[0]
    N=img_size[1]
    print(M)
    print(N)
    for u in range(M):
        for v in range(N):
            Dk=sqrt(((u-M/2-uk)**2) + ((v-N/2-vk)**2))
            Dk_=sqrt(((u-M/2+uk)**2) + ((v-N/2+vk)**2))
            if Dk==0 or Dk_ == 0:
                H[u,v]=0.0
                continue
            x=1/(1+((D0/Dk)**(2*n)))
            y=1/(1+((D0/Dk_)**(2*n)))
            H[u,v]=x*y
    return H

def butterworth(img_size,D0,uv,n):
    fil=np.ones(img_size,dtype=np.float32)
    #filt=np.ones(img_size,dtype=np.float32)
    for u,v in uv:
        temp=point_op(img_size,u,v,D0,n)
        fil*=temp
    return fil

img_input = cv2.imread('two_noise.jpg', 0)
cv2.imshow("input", img_input)
ft=np.fft.fft2(img_input)
ft_shift=np.fft.fftshift(ft)
magnitude_ac=np.abs(ft_shift)
phase=np.angle(ft_shift)
magnitude=20*np.log(magnitude_ac+1)
magnitude_n=cv2.normalize(magnitude, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
cv2.imshow("power spectrum", magnitude_n)
phase_n=cv2.normalize(phase, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
cv2.imshow("phase", phase_n)
uv=[(10,15),(18,17)]
D0=5
n=2
notch=butterworth(img_input.shape, D0, uv,n)
notch_n=cv2.normalize(notch, None,0,1,cv2.NORM_MINMAX,dtype=cv2.CV_32F)
cv2.imshow("notch", notch_n)

mag_after=notch*magnitude
mag_after_n=cv2.normalize(mag_after, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
cv2.imshow("power spectrum after", mag_after_n)
final_result=np.multiply(mag_after,np.exp(1j*phase))
img_back=np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
img_back_n=cv2.normalize(img_back, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
cv2.imshow("output", img_back_n)
cv2.waitKey(0)
cv2.destroyAllWindows()
        
        
    