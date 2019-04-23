import numpy as np
import glob

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from model_utils import save_training_graph
from sklearn.metrics import classification_report, confusion_matrix


input_shape = (224, 224, 3)
num_classes = 7

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', padding = 'Same', input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding = 'Same',))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.40))

model.add(Conv2D(128, (3, 3), activation='relu',padding = 'Same'))
model.add(Conv2D(128, (3, 3), activation='relu',padding = 'Same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.40))

model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(.50))
model.add(Dense(512,activation='relu'))
model.add(Dropout(.25))
model.add(Dense(num_classes,activation='softmax')) #final layer with softmax activation
model.summary()

sgd = keras.optimizers.SGD()
adam = keras.optimizers.Adam()
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

datagen = ImageDataGenerator(
    rescale=1. / 255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)  # randomly flip images

train_path = 'E:/Master Project/data_split/train_dir' 
valid_path = 'E:/Master Project/data_split/val_dir'
test_path = 'E:/Master Project/data_split/test_dir'

num_train_samples = len(glob.glob(train_path+'/*/*.jpg'))
num_val_samples = len(glob.glob(valid_path+'/*/*.jpg'))
train_batch_size = 10
val_batch_size = 10
image_size = 224

train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)

train_generator = datagen.flow_from_directory(train_path,
											target_size=(224,224),
											color_mode = "rgb",
											class_mode="categorical",
											batch_size=train_batch_size)

valid_gen = datagen.flow_from_directory(valid_path,
											target_size=(224,224),
											color_mode = "rgb",
											class_mode="categorical",
											batch_size=val_batch_size)
test_gen = datagen.flow_from_directory(test_path,
											target_size=(224,224),
											color_mode = "rgb",
											class_mode="categorical",
											batch_size=1)

checkpointer = ModelCheckpoint(filepath="best_weights_my_model.hdf5", 
							   monitor = 'val_acc',
							   verbose=1, 
							   save_best_only=True)


step_size_train=train_generator.n//train_generator.batch_size
step_size_val=valid_gen.n//valid_gen.batch_size
history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    callbacks=[checkpointer],
                    validation_data=valid_gen,
                    validation_steps=step_size_val,
                    epochs=30)

scores = model.evaluate_generator(generator=test_gen,steps=test_gen.n//test_gen.batch_size)
print(str(scores[0]), str(scores[1]))
save_training_graph(history, 'Training Accuracy and Loss')

Y_pred = model.predict_generator(test_gen,test_gen.n//test_gen.batch_size)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_gen.classes, y_pred))

print('Classification Report')
target_names = ['akiec','bcc','bkl','df','mel','nv','vasc']
print(classification_report(test_gen.classes, y_pred,target_names=target_names))