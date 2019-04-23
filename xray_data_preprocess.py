import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os
import shutil
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split

def add_value_labels(ax, spacing=1):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = y_value

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.

def make_cat_folders(path):
    n = os.path.join(path, 'NORMAL')
    os.mkdir(n)
    p = os.path.join(path, 'PNEUMONIA')
    os.mkdir(p)
    
def save_imgs(save_path, imgs):
    for img_path in imgs:
        img = Image.open(img_path)
        if 'NORMAL' in img_path:
            img = img.resize((224, 224))
            img.save(save_path+'/NORMAL/'+os.path.basename(img_path))
        else:
            img = img.resize((224, 224))
            img.save(save_path+'/PNEUMONIA/'+os.path.basename(img_path))

path = 'E:/Master Project/chest_xray/'

all_imgs = glob(path+'*/*/*.jpeg')

labels = ['PNEUMONIA' if 'PNEUMONIA' in img else 'NORMAL' for img in all_imgs]


classes = ('NORMAL','PNEUMONIA',)
y_pos = np.arange(len(classes))
performance = [labels.count('NORMAL'),labels.count('PNEUMONIA')]

fig, ax1 = plt.subplots(1, 1, figsize= (10, 7))
plt.bar(y_pos, performance, align='center',color=['red','blue'])
plt.xticks(y_pos, classes)
plt.title('Image Distribution')
plt.xlabel('Classes')
plt.ylabel('Number of Images')
add_value_labels(ax1)
fig.savefig('Distribution.png')


print('NORMAL: \t',labels.count('NORMAL'))
print('PNEUMONIA: \t',labels.count('PNEUMONIA'))

x_train, x_test, y_train, y_test = train_test_split(all_imgs, labels, test_size=0.2)
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.25)

#create split folders for the data set
data_path = 'E:/Master Project/new_xray_datasplit'
#shutil.rmtree(data_path, ignore_errors=True)
os.mkdir(data_path)
# create a path to 'base_dir' to which we will join the names of the new folders
# train_dir
train_dir = os.path.join(data_path, 'train')
os.mkdir(train_dir)
make_cat_folders(train_dir)
save_imgs(train_dir, x_train)

# val_dir
val_dir = os.path.join(data_path, 'val')
os.mkdir(val_dir)
make_cat_folders(val_dir)
save_imgs(val_dir, x_validate)

#test_dir
test_dir = os.path.join(data_path, 'test')
os.mkdir(test_dir)
make_cat_folders(test_dir)
save_imgs(test_dir, x_test)



    




