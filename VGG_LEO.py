import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import Input, Model
from keras.applications.vgg16 import VGG16
import cv2
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# load data
train_images = pd.read_pickle('train_max_x')  # 50000,128,128
train_labels = pd.read_csv('train_max_y.csv')['Label']
test_images = pd.read_pickle('test_max_x')  # 10000,128,128
print('before reshape: train images shape:', train_images.shape)

# For use with categorical_crossentropy.
# The softmax layer expects a tensor of size (None, 10).
train_labels = np_utils.to_categorical(train_labels, 10)


# Show image before reshape
plt.imshow(np.array(train_images[1]))
plt.show()

# Reshape data from 50000,128,128 to 50000,128,128,3
def reshaping(data):
    dim = (128, 128)
    # convert 128x128 grayscale to 128x128 rgb channels
    def to_rgb(img):
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        img_rgb = np.asarray(np.dstack((img, img, img)), dtype=np.uint8)
        return img_rgb
    rgb_list = []
    # convert train_images data to 128x128 rgb values
    for i in range(len(train_images)):
        rgb = to_rgb(train_images[i])
        rgb_list.append(rgb)
    rgb_arr = np.stack([rgb_list], axis=4)
    rgb_arr_to_3d = np.squeeze(rgb_arr, axis=4)
    return rgb_arr_to_3d

train_images = reshaping(train_images)
print('after reshape: train images shape:', train_images.shape)

# Show image after reshape
plt.imshow(np.array(train_images[1]))
plt.show()

#------------------------------------------------
# Create model without last 3 output layers (by include_top=False)
vgg_model = VGG16(weights="imagenet",include_top=False, input_tensor= Input(shape=(128, 128, 3)))
print('\n---------original vgg model summary---------')
print(vgg_model.summary())

# Loop over all layers in the vgg model and freeze them so they will
# NOT be updated during the first training process
for layer in vgg_model.layers:
    layer.trainable = False
# Create our own last output layers
model = Flatten(name='flatten')(vgg_model.output)
model = Dense(4096, activation='relu', name='fc1')(model)
model = Dense(4096, activation='relu', name='fc2')(model)
model = Dropout(0.5)(model)
model = Dense(10, activation='softmax')(model)
# Concatenate our own last output layers to the bottom of vgg model
model_vgg_mnist = Model(inputs=vgg_model.input, outputs=model, name='vgg16')
# Print model summary
print('\n---------modified vgg model summary---------')
print(model_vgg_mnist.summary())



# Design a sgd optimizer to train the model
sgd = SGD(lr=0.01, decay=1e-5)

model_vgg_mnist.compile(loss='categorical_crossentropy',
                        optimizer=sgd,
                        metrics=['accuracy'])
model_vgg_mnist.fit(train_images,
                    train_labels,
                    epochs=15,
                    validation_split=0.15,
                    verbose=1,
                    batch_size=32)
# print accuracy
result=model_vgg_mnist.evaluate(train_images,train_labels)
print('\nTrain Accuracy:',result[1])