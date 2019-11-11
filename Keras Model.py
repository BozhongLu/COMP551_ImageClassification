import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.utils.vis_utils import plot_model
import graphviz
import tensorflow
from fastai.vision import *
import os

train_images = pd.read_pickle('train_max_x')
train_labels = pd.read_csv('train_max_y.csv')
test_images = pd.read_pickle('test_max_x')
#train_labels=train_labels.iloc[:,1]

for i in range(0,50000):
    path="/Users/link/Documents/IMAGES_TRAIN"+"/"+str(train_labels.iloc[i,1])+"/"+str(train_labels.iloc[i,0])+".png"
    plt.imsave(path,train_images[i], cmap="gray_r")
    if i%100==0:
       print(str(i) + "   finished")


for i in range(0,10000):
    path="/Users/link/Documents/IMAGES_TEST"+"/"+str(i)+".png"
    plt.imsave(path,test_images[i], cmap="gray_r")
    if i%100==0:
       print(str(i) + "   finished")


"""######       PREPROCESS          ##########"""

#from sklearn.model_selection import train_test_split
#X_train, X_val, y_train, y_val = train_test_split(train_images[0:10000], train_labels[0:10000], test_size=0.33, random_state=42)


"""######       LOAD THE DATA IN           ##########"""
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

path = "C:/Users/User/Documents/IMAGES_TRAIN"
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.2, ds_tfms=get_transforms(), size=228, num_workers=4).normalize(imagenet_stats)

data.classes

data.show_batch(rows=3, figsize=(7, 8))

"""##############################################"""
"""         FAST AI         """
"""###############################################"""

from fastai.metrics import error_rate # 1 - accuracy
learn = create_cnn(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
