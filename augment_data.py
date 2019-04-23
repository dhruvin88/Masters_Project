# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 16:16:22 2019

@author: RGB
"""

from numpy.random import seed
seed(101)
from tensorflow import set_random_seed
set_random_seed(101)
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import shutil

path='E://Master Project//Skin Cancer Dataset'
path2 = 'E://Master Project//data_split2'

# note that we are not augmenting class 'nv'
class_list = ['mel','bkl','bcc','akiec','vasc','df']

for item in class_list:
    
    # We are creating temporary directories here because we delete these directories later
    # create a base dir
    aug_dir = 'E://aug_dir'
    os.mkdir(aug_dir)
    # create a dir within the base dir to store images of the same class
    img_dir = os.path.join(aug_dir, 'img_dir')
    os.mkdir(img_dir)

    # Choose a class
    img_class = item

    # list all images in that directory
    img_list = os.listdir(path2+'/train_dir/' + img_class)

    # Copy images from the class train dir to the img_dir e.g. class 'mel'
    for fname in img_list:
            # source path to image
            src = os.path.join(path2+'/train_dir/' + img_class, fname)
            # destination path to image
            dst = os.path.join(img_dir, fname)
            # copy the image from the source to the destination
            shutil.copyfile(src, dst)


    # point to a dir containing the images and not to the images themselves
    path = aug_dir
    save_path = path2+'/train_dir/' + img_class

    # Create a data generator
    datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        #brightness_range=(0.9,1.1),
        fill_mode='nearest')

    batch_size = 50

    aug_datagen = datagen.flow_from_directory(path,
                                           save_to_dir=save_path,
                                           save_format='jpg',
                                                    target_size=(224,224),
                                                    batch_size=batch_size)



    # Generate the augmented images and add them to the training folders
    
    ###########
    
    num_aug_images_wanted = 700 # total number of images we want to have in each class
    
    ###########
    
    num_files = len(os.listdir(img_dir))
    num_batches = int(np.ceil((num_aug_images_wanted-num_files)/batch_size))

    # run the generator and create about 6000 augmented images
    for i in range(0,num_batches):

        imgs, labels = next(aug_datagen)
        
    # delete temporary directory with the raw image files
    shutil.rmtree('E://aug_dir')
    
print(len(os.listdir(path2+'/train_dir/nv')))
print(len(os.listdir(path2+'/train_dir/mel')))
print(len(os.listdir(path2+'/train_dir/bkl')))
print(len(os.listdir(path2+'/train_dir/bcc')))
print(len(os.listdir(path2+'/train_dir/akiec')))
print(len(os.listdir(path2+'/train_dir/vasc')))
print(len(os.listdir(path2+'/train_dir/df')))

