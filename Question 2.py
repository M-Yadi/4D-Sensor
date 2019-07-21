# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 11:02:45 2019

@author: Mona Yadi
"""

import numpy as np
import cv2
import random


def convolve2d(image, kernel): # For convolving the kernel and the image
    output = np.zeros_like(image)            # Convolution output
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))   # Add zero padding to the input image
    image_padded[1:-1, 1:-1] = image
    for x in range(image.shape[1]):     # Loop over every pixel of the image
        for y in range(image.shape[0]):
            output[y,x]=(kernel*image_padded[y:y+3,x:x+3]).sum()        
    return output


Image = cv2.imread('Img.jpg') # Loading an Image
Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY) # Changing colorspace to grayscale; The programm can be easily modified to work for RGB pictures too
Image=Image.astype(np.float64)
row,col = Image.shape  # Extracting its rows and culumn numbers
#-------------------
#Noise = np.random.randn(row,col) # Creating a salty noise
#Noise = Noise.reshape(row,col)    
#NoiseScale=5 #the maximum amount to increase pixels' values
#NNoise=np.absolute(Noise/Noise.max())*NoiseScale   
#N_Image = Image + .01*Image * Noise # Introducing the noise to the image
#N_Image=N_Image/N_Image.max()*255
#-------------------------
#Noise = np.random.randn(row,col)
#Noise = Noise.reshape(row,col)        
#N_Image = Image + Image *.01* Noise
#------------------------------------
#s_vs_p = 0.5
#amount = 0.004
#N_Image = np.copy(Image)
## Salt mode
#num_salt = np.ceil(amount * Image.size * s_vs_p)
#coords = [np.random.randint(0, i - 1, int(num_salt))
#    for i in Image.shape]
#N_Image[coords] = 255
#--------------------------------------- ADDing NOISES (saltNpepe)
N_Image = np.zeros(Image.shape,np.float64)
prob=0.01 
thres = 1 - prob
for i in range(Image.shape[0]):
    for j in range(Image.shape[1]):
        rdn = random.random()
        if rdn < prob:
            N_Image[i][j] = 0
        elif rdn > thres:
            N_Image[i][j] = 255
        else:
            N_Image[i][j] = Image[i][j]
    
kernel1 =1/4*(np.array([[0,1,0],[1,0,1],[0,1,0]])) # neigb~ 4 kernel 

N4_Image=np.copy(N_Image)

Con1p = convolve2d(N4_Image,kernel1) #avarage matrix
Con1 =  np.subtract(N4_Image , Con1p) #differentiating
for i in range(1,row):
        for j in range(1,col):
            if(Con1[i,j]>10 or Con1[i,j]<-10): # If it is too different
                N4_Image[i,j]=np.copy(Con1p[i,j]) #change the value

                
#-------------------------- 8-Neigbor Filteration
N8_Image=np.copy(N_Image)
kernel2 =1/9*(np.array([[1,1,1],[1,1,1],[1,1,1]]))
Con2p = convolve2d(N8_Image,kernel2) #average matrix
Con3 = N8_Image - Con2p #differentiating
for i in range(1,row-1):
        for j in range(1,col-1):
            if(Con3[i,j]>10 or Con3[i,j]<-10): #If it is too different
                # Sorting Brightnesses and their coorditions within the image
                Inf2=np.array([[N8_Image[i-1,j-1],i-1,j-1],[N8_Image[i-1,j],i-1,j],[N8_Image[i-1,j+1],i-1,j+1], \
                         [N8_Image[i,j-1],i,j-1],[N8_Image[i,j],i,j],[N8_Image[i,j+1],i,j+1], \
                         [N8_Image[i+1,j-1],i+1,j-1],[N8_Image[i+1,j],i+1,j],[N8_Image[i+1,j+1],i+1,j+1]])
                Inf2=Inf2[Inf2[:,0].argsort()].astype(np.int) # sorting the values with respect to the brightnesses
                avToSub=Inf2[2:7,0].sum()/5 #calculating the avarage of 5 remaining brightness
                N8_Image[Inf2[0,1],Inf2[0,2]]=avToSub #replacing the avarage with the corrupted pixel
                N8_Image[Inf2[1,1],Inf2[1,2]]=avToSub #replacing the avarage with the corrupted pixel
                N8_Image[Inf2[7,1],Inf2[7,2]]=avToSub #replacing the avarage with the corrupted pixel
                N8_Image[Inf2[8,1],Inf2[8,2]]=avToSub #replacing the avarage with the corrupted pixel



# Preparation for depicting images (Stacking)
cv2.imshow('Image',np.vstack((np.hstack((Image.astype(np.uint8), N_Image.astype(np.uint8))),
                              np.hstack((N4_Image.astype(np.uint8), N8_Image.astype(np.uint8))))))
cv2.waitKey(0)
cv2.destroyAllWindows()

