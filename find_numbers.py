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


example=train_images[0]
plt.imshow(np.array(example), cmap='gray_r')
plt.show()

training= np.zeros([3,50000,28,28])

for i in range(0,len(train_images)):
    # define white-black threshhold
    thresh = cv2.threshold(train_images[i], 230, 255,cv2.THRESH_BINARY)[1]
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 3))
    #thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh= thresh.astype('uint8')

    #plt.imshow(thresh, cmap="gray_r")
    #plt.show()


    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    digitCnts = []
    digitw = []

    # loop over the digit area candidates
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # if the contour is sufficiently large, it must be a digit
        #if w >= 8 and (h >= 15 and h <= 40):
        digitCnts.append(c)
        digitw.append(w*h)

    len(digitCnts)

    for n in range(0,3):
        a=np.where(np.array(digitw)==max(digitw))
        (x, y, w, h) = cv2.boundingRect(digitCnts[a[0][0]])
        if (x+x+w)/2 >=14:
            left = round((x+x+w)/2)-14
        elif (x+x+w)/2 <14:
            left = 0
        if left >100:
            left = 100

        if ((y+y+h)/2) >= 14:
            top=round((y+y+h)/2)-14
        elif ((y+y+h)/2) <14:
            top = 0
        elif ((y+y+h)/2) <14:
            top = 0

        if top>100:
            top=100

        del(digitw[a[0][0]])

        # new=train_images[1][int(bottom2):int(top2)]
        # new=new[:,int(left2):int(right2)]
        new = thresh[int(top):int(top + 28)]
        new = new[:, int(left):int(left + 28)]
        training[n][i]=new
        if i%100 ==0:
            print(str(i)+"   finished")



        #rect = Rectangle((x, y), w, h, linewidth=3, edgecolor='red', facecolor='none')
        # Create a Rectangle patch
        #rect = Rectangle((left,top),28,28,linewidth=3,edgecolor='red',facecolor='none')

        #plt.imshow(thresh, cmap="gray_r")
        # Get the current reference
        #ax = plt.gca()

        # Add the patch to the Axes
        #ax.add_patch(rect)
        #ax.add_patch(rect2)
        #plt.show()


        #new.shape
        #plt.imshow(np.array(new), cmap='gray_r')
        #plt.show()


plt.imshow(training[0][1760] , cmap="gray_r")
plt.show()




for i in range(1,28):
    for  j in range(1,28):
        if i%2 == 0:
            new[i,j] = 0
        else:
            new[i, j] = 255






for contour in digitCnts:
   cv2.drawContours(np.array(thresh), contour, 1, (0, 255, 0), 3)
plt.show()
