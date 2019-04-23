# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:18:08 2019

@author: RGB
"""

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
import cv2

path='E://Master Project//new_xray_datasplit'

base_xray_dir = path+"//"

all_imgs = [x for x in glob(os.path.join(base_xray_dir, '*/*', '*.jpeg'))]

labels = ['PNEUMONIA' if 'PNEUMONIA' in img else 'NORMAL' for img in all_imgs]

imgs = [cv2.imread(img).flatten() for img in all_imgs]


x_train, x_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.2)

#loading and resixing images


print('Decision Tree Classifier')
dt = DecisionTreeClassifier(random_state=8)
dt.fit(x_train,y_train)

y_pred_test = dt.predict(x_test)
  
print("Test data metrics:")
print(sklearn.metrics.classification_report(y_true= y_test, y_pred= y_pred_test))


print('Random Forest Classifier')
rf = RandomForestClassifier(n_estimators=100, random_state=8)
rf.fit(x_train, y_train)

y_pred_test = rf.predict(x_test)

print("Test data metrics:")
print(sklearn.metrics.classification_report(y_true= y_test, y_pred= y_pred_test))
