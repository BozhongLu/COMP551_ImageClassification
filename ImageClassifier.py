import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

train_images = pd.read_pickle('train_max_x')
test_images = pd.read_pickle('test_max_x')
train_labels = pd.read_csv('train_max_y.csv')

plt.title('Label: {}'.format(train_labels.iloc[train_images[0]]['Label']))
plt.imshow(np.array(train_images[0]), cmap='gray', vmin=0, vmax=255)
plt.show()

