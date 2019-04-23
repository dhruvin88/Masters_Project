# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 17:55:10 2019

@author: RGB
"""

import numpy as np
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.applications import VGG16, ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import keras.optimizers 
from model_utils import save_training_graph, plot_confusion_matrix, vis_activation, get_img_fit_flow
from sklearn.metrics import classification_report, confusion_matrix
from keras.applications.vgg16 import preprocess_input

model_path = 'E:/Drive/Grad School/Master Project/Unfreeze Beginning Layers/best_weights_model.hdf5'
test_path = 'E:/Master Project/data_split/test_dir'

#load model
model = load_model(model_path)

test_datagen=ImageDataGenerator(
        preprocessing_function=preprocess_input)

test_gen = test_datagen.flow_from_directory(test_path,
                                            target_size=(224,224),
                                            color_mode = "rgb",
                                            class_mode="categorical",
                                            batch_size=1,
                                            shuffle=False)

Y_pred = model.predict_generator(test_gen,test_gen.n//test_gen.batch_size)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
confusion_mtx=confusion_matrix(test_gen.classes, y_pred)
print(confusion_mtx)
plot_confusion_matrix(confusion_mtx, classes = range(7))

print('Classification Report')
target_names = ['akiec','bcc','bkl','df','mel','nv','vasc']
print(classification_report(test_gen.classes, y_pred,target_names=target_names))

