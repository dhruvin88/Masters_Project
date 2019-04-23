# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 18:05:33 2019

@author: RGB
"""

import numpy as np
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import keras.optimizers 
from model_utils import save_training_graph, plot_confusion_matrix, vis_activation
from sklearn.metrics import classification_report, confusion_matrix
from keras.applications.vgg16 import preprocess_input

base_model=VGG16(weights=None,include_top=False, input_shape=(224,224,3))

top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(1024,activation='relu'))
top_model.add(Dropout(.50))
top_model.add(Dense(512,activation='relu'))
top_model.add(Dropout(.25))

top_model.add(Dense(7,activation='softmax')) #final layer with softmax activation

model=Model(inputs=base_model.input,outputs=top_model(base_model.output))
print(model.summary())

train_datagen=ImageDataGenerator(
    preprocessing_function=preprocess_input,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    horizontal_flip=True,
    fill_mode='nearest')

valid_datagen=ImageDataGenerator(
        preprocessing_function=preprocess_input)

train_path = 'E:/Master Project/data_split/train_dir' 
valid_path = 'E:/Master Project/data_split/val_dir' 
test_path = 'E:/Master Project/data_split/test_dir' 

train_generator = train_datagen.flow_from_directory(train_path,
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=10,
                                                 class_mode='categorical',
                                                 shuffle=True)

valid_gen = valid_datagen.flow_from_directory(valid_path,
                                            target_size=(224,224),
                                            color_mode = "rgb",
                                            class_mode="categorical",
                                            batch_size=10)


# loss function will be categorical cross entropy
# evaluation metric will be accuracy
sgd = keras.optimizers.SGD()
adam = keras.optimizers.Adam()
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              verbose=0, mode='auto')

checkpointer = ModelCheckpoint(filepath="best_weights_model.hdf5", 
                               monitor = 'val_acc',
                               verbose=1, 
                               save_best_only=True)

step_size_train=train_generator.n//train_generator.batch_size
step_size_val=valid_gen.n//valid_gen.batch_size

history = model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   callbacks=[learning_rate_reduction,early_stop],
                   validation_data=valid_gen,
                   validation_steps=step_size_val,
                   epochs=50)

base_model=VGG16(weights='imagenet',include_top=False, input_shape=(224,224,3))

top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(1024,activation='relu'))
top_model.add(Dropout(.50))
top_model.add(Dense(512,activation='relu'))
top_model.add(Dropout(.25))

top_model.add(Dense(7,activation='softmax')) #final layer with softmax activation

model=Model(inputs=base_model.input,outputs=top_model(base_model.output))
print(model.summary())
sgd = keras.optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
history2 = model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   callbacks=[learning_rate_reduction,early_stop],
                   validation_data=valid_gen,
                   validation_steps=step_size_val,
                   epochs=50)

import matplotlib.pyplot as plt

plt.plot(history.history['val_acc'], color='b', linestyle='solid', label="Without ImageNet Weights")
plt.plot(history2.history['val_acc'], color='r', linestyle='dashed', label="With ImageNet Weights")
legend = plt.legend(loc='best', shadow=True)
plt.savefig('accuracy_with_and_without.png')

plt.plot(history.history['acc'], color='b',linestyle='solid', label="Without ImageNet Weights")
plt.plot(history2.history['acc'], color='r',linestyle='dashed',label="With ImageNet Weights")
legend = plt.legend(loc='lower right', shadow=True)
plt.savefig('accuracy_with_and_without2.png')


