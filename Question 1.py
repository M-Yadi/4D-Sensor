# -*- coding: utf-8 -*-
"""
Created on Fri July 19 12:19:54 2019
@author: Mona Yadi
"""
import numpy as np
import cv2

Image_size=500
Grating_pitch=20
Grating1D=np.zeros((500,500))
Grating2D=np.zeros((500,500))


#----------------------------------Cos waves / Soft edges
t= np.arange(0,Image_size,1)  
a= np.cos(2*np.pi*1/Grating_pitch*t)*255/2 #1D
A=a+255/2
for i in t:
   Grating1D[i,:]=A

Grating2D=np.multiply(Grating1D,np.transpose(Grating1D)); # To obtain 2D pattern 
Grating2D=Grating2D/Grating2D.max()*255 # Normalization and rescaling to 0-255

#------------------Hard edges---------- As question #1 does not describe that if it needs hard or soft edges / I have considered hard edges too
Grating_pitch=20
Image_size=500

a=np.ones((Image_size,Image_size))
y=np.arange(0,np.ceil(Image_size/Grating_pitch),1).astype(int)
for h in y:
    a[:,h*20:h*20+10]=0
c=np.transpose(a)
g=np.multiply(a,c); g=g*255
a=a*255


cv2.imshow('Gratings',np.hstack((Grating1D.astype(np.uint8), Grating2D.astype(np.uint8))))
cv2.imshow('Images',np.hstack((a.astype(np.uint8), g.astype(np.uint8))))
cv2.waitKey(0)
cv2.destroyAllWindows()