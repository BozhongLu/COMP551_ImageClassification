import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

train_images = pd.read_pickle('train_max_x')
test_images = pd.read_pickle('test_max_x')


from PIL import Image


plt.imshow(np.array(train_images[0]), cmap='gray', vmin=0, vmax=255)
plt.show()