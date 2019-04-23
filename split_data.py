import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image
import scipy.misc

from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding

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


# get dictionary with all file name in .jpg
base_dir = '../../../Master Project/Skin Cancer Dataset/'

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_dir, '*', '*.jpg'))}
print(len(imageid_path_dict))

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

# Read in the csv of metadata
tile_df = pd.read_csv(os.path.join(base_dir, 'HAM10000_metadata.csv'))

# Create some new columns (path to image, human-readable name) and review them
tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get)
tile_df['cell_type'] = tile_df['dx'].map(lesion_type_dict.get) 
tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes

# distribution of different cell types
fig, ax1 = plt.subplots(1, 1, figsize= (10, 7))
tile_df['dx'].value_counts().plot(kind='bar', ax=ax1)
plt.title('Image Distribution')
plt.xlabel('Classes')
plt.ylabel('Number of Images')
add_value_labels(ax1)

fig.savefig('Distribution.png')

# Too many melanocytic nevi - let's balance it a bit!
#tile_df = tile_df.drop(tile_df[tile_df.cell_type_idx == 4].iloc[:5000].index)
'''
fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))
tile_df['cell_type'].value_counts().plot(kind='bar', ax=ax1)
fig.savefig('Changed Distribution.png')
'''
# Load in all of the images into memory - this will take a while.  
# We also do a resize step because the original dimensions of 450 * 600 * 3 was too much for TensorFlow
#tile_df['image'] = tile_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))
tile_df['image'] = tile_df['path'].map(lambda x: np.asarray(Image.open(x).resize((224,224))))

n_samples = 5
fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
for n_axs, (type_name, type_rows) in zip(m_axs, 
                                         tile_df.sort_values(['cell_type']).groupby('cell_type')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=2018).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
fig.savefig('category_samples.png', dpi=300)

y = tile_df.cell_type_idx

#make training, validation, and test split of the data
from sklearn.model_selection import train_test_split
x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(tile_df, y, test_size=0.2)

#x_train = np.asarray(x_train_o['image'].tolist())
#x_test = np.asarray(x_test_o['image'].tolist())

#x_train_mean = np.mean(x_train)
#x_train_std = np.std(x_train)

#x_test_mean = np.mean(x_test)
#x_test_std = np.std(x_test)

#x_train = (x_train - x_train_mean)/x_train_std
#x_test = (x_test - x_test_mean)/x_test_std

# Perform one-hot encoding on the labels
y_train = to_categorical(y_train_o, num_classes = 7)
y_test = to_categorical(y_test_o, num_classes = 7)

x_train, x_validate, y_train, y_validate = train_test_split(x_train_o, y_train, test_size = 0.25)
'''
x_train = x_train.reshape(x_train.shape[0], *(224, 224, 3))
x_test = x_test.reshape(x_test.shape[0], *(224, 224, 3))
x_validate = x_validate.reshape(x_validate.shape[0], *(224, 224, 3))
'''

#create split folders for the data set
data_path = 'E:/Master Project/data_split2'
# create a path to 'base_dir' to which we will join the names of the new folders
# train_dir
train_dir = os.path.join(data_path, 'train_dir')
os.mkdir(train_dir)

# val_dir
val_dir = os.path.join(data_path, 'val_dir')
os.mkdir(val_dir)

#test_dir
test_dir = os.path.join(data_path, 'test_dir')
os.mkdir(test_dir)

def make_cat_folders(path):
    nv = os.path.join(path, 'nv')
    os.mkdir(nv)
    mel = os.path.join(path, 'mel')
    os.mkdir(mel)
    bkl = os.path.join(path, 'bkl')
    os.mkdir(bkl)
    bcc = os.path.join(path, 'bcc')
    os.mkdir(bcc)
    akiec = os.path.join(path, 'akiec')
    os.mkdir(akiec)
    vasc = os.path.join(path, 'vasc')
    os.mkdir(vasc)
    df = os.path.join(path, 'df')
    os.mkdir(df)
    
make_cat_folders(train_dir)
make_cat_folders(val_dir)
make_cat_folders(test_dir)

def save_split(path, data):
    images = data['image'].tolist()
    image_id = data['image_id'].tolist()
    image_dx = data['dx'].tolist()
    
    for i in range(len(images)):
        images[i] = images[i]*(1/255)
    
    for i in range(len(images)):
        scipy.misc.imsave((path+'/'+image_dx[i]+'/'+image_id[i]+'.jpg'), images[i])

save_split(train_dir, x_train)
save_split(test_dir, x_test_o)
save_split(val_dir, x_validate)

'''
x_train =  x_train['image'].tolist()
x_test = x_test_o['image'].tolist()
x_validate = x_validate['image'].tolist()

x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)

x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)

x_val_mean = np.mean(x_validate)
x_val_std = np.std(x_validate)

x_train = (x_train - x_train_mean)/x_train_std
x_test = (x_test - x_test_mean)/x_test_std
x_validate = (x_validate - x_val_mean)/x_val_std

# save data split
np.save('x_train', x_train)
np.save('x_test', x_test)
np.save('x_validate', x_validate)

np.save('y_train', y_train)
np.save('y_test', y_test)
np.save('y_validate', y_validate)

'''
