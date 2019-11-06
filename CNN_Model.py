import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import datasets
from keras import layers
from keras import models



# Build CNN class
class CNN(object):
    def __init__(self):
        model = models.Sequential()
        # first layer, 32 kernels with kernel size 3*3 , 128*128*1 image size from processed data
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        # second layer, 64 kernels with kernel size 3*3
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        # third layer, 64 kernels with kernel size 3*3
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        # print the structure of the model
        model.summary()
        self.model = model

class Data(object):
    def __init__(self):
        # load data
        train_images = pd.read_pickle('train_max_x')  # 50000,128,128
        train_labels = pd.read_csv('train_max_y.csv')['Label']
        test_images = pd.read_pickle('test_max_x')  # 10000,128,128
        # Reshape images
        train_images = train_images.reshape((50000, 128, 128, 1))
        test_images = test_images.reshape((10000, 128, 128, 1))
        # 像素值映射到 0 - 1 之间
        train_images, test_images = train_images / 255.0, test_images / 255.0

        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
#       self.test_labels = test_labels

class Train:
    def __init__(self):
        self.cnn = CNN()
        self.data = Data()

    def train(self):
        self.cnn.model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        self.cnn.model.fit(self.data.train_images, self.data.train_labels, epochs = 5)

        # after fitting the model, test the model accuracy
        test_loss, test_acc = self.cnn.model.evaluate(self.data.train_labels, self.data.train_labels)
        print("Accuracy: %.4f" % test_acc)

if __name__ == '__main__':
    app = Train()
    app.train()