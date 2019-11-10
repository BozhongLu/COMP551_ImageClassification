import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from keras.utils.vis_utils import plot_model
import graphviz
import tensorflow

train_images = pd.read_pickle('train_max_x')
train_labels = pd.read_csv('train_max_y.csv')
train_labels=train_labels.iloc[:,1]


"""######       PREPROCESS          ##########"""
# convert 128x128 grayscale to 124x124 rgb channels
dim = (124, 124)

def to_rgb(img):
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    img_rgb = np.asarray(np.dstack((img, img, img)), dtype=np.uint8)
    return img_rgb

rgb_list = []
# convert X_train data to 48x48 rgb values
for i in range(len(train_images)):
    rgb = to_rgb(train_images[i])
    rgb_list.append(rgb)
    # print(rgb.shape)
    if i%100 ==0:
        print(str(i) + "   finished")


rgb_arr = np.stack([rgb_list], axis=4)
rgb_arr_to_3d = np.squeeze(rgb_arr, axis=4)
print(rgb_arr_to_3d.shape)


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(rgb_arr_to_3d, train_labels, test_size=0.33, random_state=42)


"""######       PREPARE MODEL VGG16          ##########"""

vgg_conv = VGG16(weights='imagenet', include_top=False,input_shape=(124, 124, 3))

# pooling
print(vgg_conv.summary())
#plot_model(model, to_file='vgg.png')

# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-1]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)

# add new output layer
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

# Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(vgg_conv)

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()


"""######       LOAD THE DATA INTO           ##########"""
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

BATCH_SIZE = 100
SHUFFLE_BUFFER_SIZE = 10

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)


"""######       COMPILE MODEL           ##########"""

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# Train the model
history = model.fit_generator(
    train_dataset,
    epochs=3,
    validation_data=val_dataset,
    verbose=1)

# Save the model
#model.save('small_last4.h5')
