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
