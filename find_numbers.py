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

# define white-black threshhold
thresh = cv2.threshold(example, 150, 255,cv2.THRESH_BINARY)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 3))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

plt.imshow(np.array(thresh), cmap='gray_r')
plt.show()

thresh= thresh.astype('uint8')

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
digitCnts = []

# loop over the digit area candidates
for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)

    # if the contour is sufficiently large, it must be a digit
    if w >= 10 and (h >= 20 and h <= 40):
        digitCnts.append(c)


for i in range(0,len(digitCnts)):
    left = np.min(digitCnts[i][:,0][:,0])
    top = np.max(digitCnts[i][:,0][:,1])
    right = np.max(digitCnts[i][:,0][:,0])
    bottom = np.min(digitCnts[i][:,0][:,1])

    left2 = round((left+right)/2)-14
    right2 = round((left+right)/2)+14
    top2=round((bottom+top)/2)+14
    bottom2 = round((bottom + top) / 2) - 14

    # Create a Rectangle patch
    rect = Rectangle((left,top),right-left,bottom-top,linewidth=3,edgecolor='red',facecolor='none')
    rect2 = Rectangle((left2,top2),right2-left2,bottom2-top2,linewidth=3,edgecolor='blue',facecolor='none')

    plt.imshow(thresh, cmap="gray_r")
    # Get the current reference
    ax = plt.gca()

    # Add the patch to the Axes
    ax.add_patch(rect)
    ax.add_patch(rect2)
plt.show()

#new=train_images[1][int(bottom2):int(top2)]
#new=new[:,int(left2):int(right2)]
new=thresh[int(bottom2):int(top2)]
new=new[:,int(left2):int(right2)]
new.shape
plt.imshow(np.array(new), cmap='gray_r')
plt.show()


for contour in digitCnts:
   cv2.drawContours(np.array(thresh), contour, 1, (0, 255, 0), 3)
plt.show()
