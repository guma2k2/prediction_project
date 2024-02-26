import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# libraries for system
import os
import time
import shutil
import pathlib
import itertools
from PIL import Image


# import data handling tools
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report



#import DL libraries
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
# from tensorflow.python.keras.optimizers import adamax_v2
from keras.optimizers import Adamax
from tensorflow.python.keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Conv2D, Dense, BatchNormalization, Activation, Dropout, MaxPooling2D, Flatten
from tensorflow.python.keras import regularizers

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")
import os


# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


#  Data preprocessing
train_dir = '/Applications/workspace/python_project/prediction_project/archive/train'
filepaths = []
labels = []

folds = os.listdir(train_dir)
for fold in folds:
    foldpath = os.path.join(train_dir, fold)
    if os.path.isdir(foldpath):
        filelist = os.listdir(foldpath)
        
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            
            # Check if the item is a file
            if os.path.isfile(fpath):
                filepaths.append(fpath)
                labels.append(fold)

# Concatenate data paths with labels into one dataframe
Fseries = pd.Series(filepaths, name= 'filepaths')
Lseries = pd.Series(labels, name='labels')
train_df = pd.concat([Fseries, Lseries], axis= 1)


test_dir = '/Applications/workspace/python_project/prediction_project/archive/test'
filepaths = []
labels = []

folds = os.listdir(test_dir)
for fold in folds:
    foldpath = os.path.join(test_dir, fold)
    if os.path.isdir(foldpath):
        filelist = os.listdir(foldpath)
        
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            
            # Check if the item is a file
            if os.path.isfile(fpath):
                filepaths.append(fpath)
                labels.append(fold)

# Concatenate data paths with labels into one dataframe
Fseries = pd.Series(filepaths, name= 'filepaths')
Lseries = pd.Series(labels, name='labels')
test_df = pd.concat([Fseries, Lseries], axis= 1)

valid_df, test_df = train_test_split(test_df,  train_size= 0.6, shuffle= True, random_state= 123)

batch_size = 16
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)

tr_gen = ImageDataGenerator()
ts_gen = ImageDataGenerator()
train_gen = tr_gen.flow_from_dataframe( train_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= True, batch_size= batch_size)

valid_gen = ts_gen.flow_from_dataframe( valid_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= True, batch_size= batch_size)

test_gen = ts_gen.flow_from_dataframe( test_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= False, batch_size= batch_size)

#Showing sample from our train data
# g_dict = train_gen.class_indices      # defines dictionary {'class': index}
# classes = list(g_dict.keys())       # defines list of dictionary's kays (classes), classes names : string
# images, labels = next(train_gen)      # get a batch size samples from the generator

# plt.figure(figsize= (20, 20))

# for i in range(16):
#     plt.subplot(4, 4, i + 1)
#     image = images[i] / 255       # scales data to range (0 - 255)
#     plt.imshow(image)
#     index = np.argmax(labels[i])  # get image index
#     class_name = classes[index]   # get class of image
#     plt.title(class_name, color= 'blue', fontsize= 12)
#     plt.axis('off')
# plt.show()


img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)
class_count = len(list(train_gen.class_indices.keys())) # to define number of classes in dense layer

# create pre-trained model (you can built on pretrained model such as :  efficientnet, VGG , Resnet )
# we will use efficientnetb3 from EfficientNet family.
base_model = tf.keras.applications.efficientnet.EfficientNetB7(include_top= False, weights= "imagenet", input_shape= img_shape, pooling= 'max')
# base_model.trainable = False
model = Sequential([
    base_model,
    BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
    Dense(256, kernel_regularizer= regularizers.l2(l= 0.016), activity_regularizer= regularizers.l1(0.006),
                bias_regularizer= regularizers.l1(0.006), activation= 'relu'),
    Dropout(rate= 0.45, seed= 123),
    Dense(class_count, activation= 'softmax')
])
model.compile(Adamax(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

batch_size = 20   # set batch size for training
epochs =  1 # number of all epochs in training

history = model.fit(x= train_gen, epochs= epochs, verbose= 1, validation_data= valid_gen, 
                    validation_steps= None, shuffle= False)


model.save("model.h5")


# image_path2 = '/kaggle/input/emotion-detection-fer/test/surprised/im1.png'
# image = Image.open(image_path2)

# # Preprocess the image
# img = image.resize((224, 224))
# img_array = tf.keras.preprocessing.image.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0)

# # Make predictions
# predictions = loaded_model.predict(img_array)
# class_labels = classes
# score = tf.nn.softmax(predictions[0])
# print(f"{class_labels[tf.argmax(score)]}")
