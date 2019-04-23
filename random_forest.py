# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 10:40:55 2019

@author: RGB
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


path='E://Master Project//Skin Cancer Dataset'

base_skin_dir = path+"//"

# Merging images from both folders HAM10000_images_part1.zip and HAM10000_images_part2.zip into one dictionary

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}

# This dictionary is useful for displaying more human-friendly labels later on

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

skin_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))

# Creating New Columns for better readability

skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get) 
skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes

#loading and resixing images
skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((224,224))).flatten())
features=skin_df.drop(columns=['cell_type_idx'],axis=1)
target=skin_df.dx.astype('str')

x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.20,random_state=1234)
x_train = np.asarray(x_train_o['image'].tolist())
x_test = np.asarray(x_test_o['image'].tolist())

#normalize
'''
x_train_mean = np.mean(x_train)  #=~159
x_train_std = np.std(x_train)    #=~47

x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)

x_train = (x_train - 159)/47
x_test = (x_test - 159)/47
'''

#rescale
'''
x_train = x_train / 127.5 - 1
x_test = x_test / 127.5 - 1
'''

print('Decision Tree Classifier')
dt = DecisionTreeClassifier(random_state=8)
dt.fit(x_train,y_train_o)

y_pred_test = dt.predict(x_test)
  
print("Test data metrics:")
print(sklearn.metrics.classification_report(y_true= y_test_o, y_pred= y_pred_test))


print('Random Forest Classifier')
rf = RandomForestClassifier(n_estimators=100, random_state=8)
rf.fit(x_train, y_train_o)

y_pred_test = rf.predict(x_test)

print("Test data metrics:")
print(sklearn.metrics.classification_report(y_true= y_test_o, y_pred= y_pred_test))
