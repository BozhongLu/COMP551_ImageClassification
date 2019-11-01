import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#
train_images = pd.read_pickle('train_max_x')
test_images = pd.read_pickle('test_max_x')
train_labels = pd.read_csv('train_max_y.csv')

# choose a file, run the cleaning code and import them here
cleaned_train=pd.read_pickle("cleaned_train")
cleaned_test=pd.read_pickle("cleaned_test")

#choose image to plot
imageplot=11

fig=plt.figure(figsize=(8, 8))
fig.add_subplot(1, 2, 1)
plt.imshow(np.array(cleaned_train[imageplot]), cmap='gray_r')
plt.title('Label: {}'.format(train_labels.iloc[imageplot]['Label']))
fig.add_subplot(1, 2, 2)

plt.imshow(np.array(train_images[imageplot]), cmap='gray_r')
plt.title('Label: {}'.format(train_labels.iloc[imageplot]['Label']))
plt.show()