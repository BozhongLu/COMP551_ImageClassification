import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from imutils import contours
import imutils
import cv2
from matplotlib.patches import Rectangle
from PIL import Image


train_images = pd.read_pickle('train_max_x')
test_images = pd.read_pickle('test_max_x')

cleaned_train=pd.read_pickle("cleaned_train")
cleaned_test=pd.read_pickle("cleaned_test")

example=train_images[1]
plt.imshow(np.array(example), cmap='gray_r')
plt.show()


thresh = cv2.threshold(example, 180, 255,cv2.THRESH_BINARY)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 3))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


thresh= thresh.astype('uint8')

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
digitCnts = []

# loop over the digit area candidates
for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)

    # if the contour is sufficiently large, it must be a digit
    if w >= 10 and (h >= 10 and h <= 40):
        digitCnts.append(c)

for i in len(digitCnts):

    left = np.min(digitCnts[1][:,0][:,0])
    top = np.max(digitCnts[1][:,0][:,1])
    right = np.max(digitCnts[1][:,0][:,0])
    bottom = np.min(digitCnts[1][:,0][:,1])

    plt.imshow(thresh,cmap="gray")
    # Get the current reference
    ax = plt.gca()
    # Create a Rectangle patch
    rect = Rectangle((left,top),right-left,bottom-top,linewidth=1,edgecolor='r',facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)

plt.show()

for contour in digitCnts:
   cv2.drawContours(np.array(thresh), contour, 1, (0, 255, 0), 3)
plt.show()
