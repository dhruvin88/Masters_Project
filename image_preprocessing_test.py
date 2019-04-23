# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 13:57:03 2019

@author: RGB
"""

import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt


def normalize_color_image(img_path):
    '''
    img = cv2.imread(img_path)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    
    #equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    
    #convert the YUV image back to RGB 
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    return img_output
    '''
    image = cv2.imread(img_path)
    channels = cv2.split(image)
    eq_channels = []
    for ch, color in zip(channels, ['B', 'G', 'R']):
        eq_channels.append(cv2.equalizeHist(ch))

    eq_image = cv2.merge(eq_channels)
    eq_image = cv2.cvtColor(eq_image, cv2.COLOR_BGR2RGB)
    return eq_image
    
    


train_path = 'E:/Master Project/data_split3/train_dir' 
valid_path = 'E:/Master Project/data_split3/val_dir' 
test_path = 'E:/Master Project/data_split3/test_dir'

train_pics = glob(train_path+"/*/*.jpg")
val_pics = glob(valid_path+"/*/*.jpg")
test_pics = glob(test_path+"/*/*.jpg")

for pic in train_pics:
    img = normalize_color_image(pic)
    cv2.imwrite(pic, img)
for pic in val_pics:
    img = normalize_color_image(pic)
    cv2.imwrite(pic, img)
for pic in test_pics:
    img = normalize_color_image(pic)
    cv2.imwrite(pic, img)




