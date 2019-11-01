from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
import pandas as pd
import os
import numpy as np
from sklearn import metrics

train_images = pd.read_pickle('train_max_x')
test_images = pd.read_pickle('test_max_x')
train_labels = pd.read_csv('train_max_y.csv')

