import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import datasets
from keras import layers
from keras import models


# Input image data
train_images = pd.read_pickle('train_max_x')
test_images = pd.read_pickle('test_max_x')
# Show original sample
example=train_images[1761]
plt.imshow(np.array(example), cmap='gray_r')
plt.show()

# Build CNN class
class CNN(object):
    def __init__(self):
        model = models.Sequential()
        # first layer, 32 kernels with kernel size 3*3 , 28*28 image size
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        # second layer, 64 kernels with kernel size 3*3
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        # third layer, 64 kernels with kernel size 3*3
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        model.summary()

        self.model = model
