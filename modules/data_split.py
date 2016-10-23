'''

   This module performs the function of splitting the training data into
   two datasets, for testing out the system on the second dataset (test *)

   Usage: import data_split as split
           test_size = 0.2
           split.split_data(test_size)

   Returns: A list of four datasets. X_train, y_train, X_test, y_test in this
           order.

'''

import numpy as np
import constants as c
from sklearn.cross_validation import train_test_split as tts

# Load training data (images and their masks)
def load_train_data():
    imgs_train = np.load(c.NPY_PATH + c.TRAIN_NPY)
    imgs_mask_train = np.load(c.NPY_PATH + c.TRAIN_MASKS_NPY)
    return [imgs_train, imgs_mask_train]

# Split training data
def split_data(fraction):
    imgs_train, imgs_mask_train = load_train_data()
    return tts(imgs_train, imgs_mask_train, test_size = fraction, random_state = 0)
