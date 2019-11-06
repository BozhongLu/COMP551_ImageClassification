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


example=train_images[4000]
plt.imshow(np.array(example), cmap='gray_r')
plt.show()

# Function that recognize digits in an image and rotate the ones with angle to a correct angle
#Input should be train_images, test_images
def imagePreprocessing(imageSet):

    testing = np.zeros([3, 50000, 28, 28])
    #test = np.zeros([3, 50000, 28, 28])
    output = np.zeros([3, 50000, 28, 28])

    for i in range(0,len(imageSet)):
        # define white-black threshhold
        thresh = cv2.threshold(imageSet[i], 230, 255,cv2.THRESH_BINARY)[1]
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 3))
        #thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh= thresh.astype('uint8')


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

            # cut the digits area out from the preprocessed images
            new = thresh[int(top):int(top + 28)]
            new = new[:, int(left):int(left + 28)]
            # Save for comparison
            testing[n][i] = new
            #Rotate the image to a correct angle to improve accuracy
            coords = np.column_stack(np.where(new > 0))
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
               angle = -(90 + angle)
            else:
               angle = -angle

            (h, w) = new.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(new, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            #Define a clearer threshold for the preprocessed images
            newthresh = cv2.threshold(rotated, 100, 255, cv2.THRESH_BINARY)[1]
            output[n][i]=newthresh
            if i%100 ==0:
                print(str(i)+"   finished")
    #Testing
    plt.imshow(testing[0][4000], cmap="gray_r")
    plt.show()

    plt.imshow(testing[1][4000], cmap="gray_r")
    plt.show()

    plt.imshow(testing[2][4000], cmap="gray_r")
    plt.show()

    plt.imshow(output[0][4000] , cmap="gray_r")
    plt.show()

    plt.imshow(output[1][4000] , cmap="gray_r")
    plt.show()

    plt.imshow(output[2][4000], cmap="gray_r")
    plt.show()

imagePreprocessing(train_images)

