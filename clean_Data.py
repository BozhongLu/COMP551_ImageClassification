import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# install___  pip install opencv-python

train_images = pd.read_pickle('train_max_x')
test_images = pd.read_pickle('test_max_x')

cleaned_train=pd.read_pickle("C:/Users/User/Documents/2_Programming/Machine_Learning/COMP 551/Project3/cleaned_train")
cleaned_test=pd.read_pickle("C:/Users/User/Documents/2_Programming/Machine_Learning/COMP 551/Project3/cleaned_test")

def remove_background(T, image):
    # grab the image dimensions
    h = image.shape[0]
    w = image.shape[1]

    # loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            # threshold the pixel
            image[y, x] = 255 if image[y, x] >= T else 0

    # return the thresholded image
    return image

for i in range(0,cleaned_train.shape[0]):
    cleaned_train[i]=remove_background(230,cleaned_train[i])
    if i%100==0:
        print(str(i)+"  done")

for i in range(0,cleaned_test.shape[0]):
    cleaned_test[i]=remove_background(230,cleaned_test[i])
    if i%100==0:
        print(str(i)+"  done")

"""choose your saving file """
pd.to_pickle(cleaned_train,"file")
pd.to_pickle(cleaned_test, "file")
