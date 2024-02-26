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


loaded_model = tf.keras.models.load_model('/Applications/workspace/python_project/prediction_project/model.h5', compile=False)
loaded_model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])


image_path = '/Applications/workspace/python_project/prediction_project/archive/test/happy/im3.png'
image = Image.open(image_path)

# Preprocess the image
img = image.resize((224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

# Make predictions
predictions = loaded_model.predict(img_array)
class_labels = ["angry", "happy", "sad"]
score = tf.nn.softmax(predictions[0])
print(f"{class_labels[tf.argmax(score)]}")
