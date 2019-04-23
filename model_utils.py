# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 11:09:48 2018

@author: RGB
"""

import matplotlib.pyplot as plt
import numpy as np
import itertools

from keras import models
from keras.preprocessing import image


def save_training_graph(history, filename):
    fig, ax = plt.subplots(2,1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="Validation loss",axes =ax[0])
    legend = ax[0].legend(loc='best', shadow=True)
    
    ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    fig.savefig(filename)
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('Confusion_Matrix.png')
    
def vis_activation(model, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    
    plt.imshow(img_tensor[0])
    plt.show()
    print(img_tensor.shape)
    
    layer_outputs = [layer.output for layer in model.layers[1:19]] 
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs) 
    # Creates a model that will return these outputs, given the model input
    
    activations = activation_model.predict(img_tensor)
    
    layer_names = []
    for layer in model.layers[1:19]:
    	layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
    	
    images_per_row = 16
    
    for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    	n_features = layer_activation.shape[-1] # Number of features in the feature map
    	width = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    	height = layer_activation.shape[2]
    	n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    	display_grid = np.zeros((width * n_cols, images_per_row * height))
    	
    	for col in range(n_cols): # Tiles each filter into a big horizontal grid 
    		for row in range(images_per_row):
    			channel_image = layer_activation[0,:, :,col * images_per_row + row]
    			channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
    			channel_image /= channel_image.std()
    			channel_image *= 64
    			channel_image += 128
    			channel_image = np.clip(channel_image, 0, 255).astype('uint8')
    			display_grid[col * width : (col + 1) * width, # Displays the grid
    						 row * height : (row + 1) * height] = channel_image
    	#scale = 1. / (224)
    	#plt.figure(figsize=(scale * display_grid.shape[1],
    	#					scale * display_grid.shape[0]))
    	plt.title(layer_name)
    	plt.grid(False)
    	plt.imshow(display_grid, aspect='auto', cmap='viridis')
    	plt.savefig('layer_'+layer_name+'.png')
        
