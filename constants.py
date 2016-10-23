#
#   This global module contains all system constants like filenames, image sizes, paths.
#   Example usage: import constants as c
#                   print (c.IMAGE_SIZE[0])
#

import os

# File names
TEST_NPY = 'imgs_test.npy'
TEST_IDS_NPY = 'imgs_id_test.npy'
TRAIN_NPY = 'imgs_train.npy'
TRAIN_MASKS_NPY = 'imgs_mask_train.npy'

# Suffixes & Prefixes
MASK_SUFFIX = '_mask.tif'

# Paths
DATA_PATH = '../data/raw/'
NPY_PATH = '../data/npy/'
TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'train')
TEST_DATA_PATH = os.path.join(DATA_PATH, 'test')

# Data Image Size
IMAGE_SIZE = [420, 580]
