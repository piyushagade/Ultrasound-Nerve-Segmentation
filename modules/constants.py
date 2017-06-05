'''

   This global module contains all system constants like filenames, image sizes, paths.

   Example usage: import constants as c
                   print (c.IMAGE_SIZE[0])

'''

import os

# File names
TEST_NPY = 'imgs_test.npy'
TEST_IDS_NPY = 'imgs_id_test.npy'
TRAIN_NPY = 'imgs_train.npy'
TRAIN_MASKS_NPY = 'imgs_mask_train.npy'
VALIDATION_NPY = 'imgs_validation.npy'
VALIDATION_MASKS_NPY = 'imgs_mask_validation.npy'

# Suffixes & Prefixes
MASK_SUFFIX = '_mask.tif'

# Paths
DATA_PATH = '../data/raw/'
NPY_PATH = '../data/npy/'
TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'train')
VALIDATION_DATA_PATH = os.path.join(DATA_PATH, 'validation')
TEST_DATA_PATH = os.path.join(DATA_PATH, 'test')

# Data Image Size
IMAGE_SIZE = [420, 580]
# Height, Width
RESIZE_FACTOR  = [1/6.5625, 1/7.25]
