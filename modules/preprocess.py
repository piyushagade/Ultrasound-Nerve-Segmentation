'''

   This module reads the contents (the ultrasound images and masks), downsizes them, converts them
   into numpy arrays and persists them in npy files for later use.

   Functions to be called: preprocess_data()
   Returns: doesn't return any object

'''

from __future__ import print_function
import os
import numpy as np
import cv2
import constants as c
from sys import stdout
from time import sleep

height = int(c.IMAGE_SIZE[0] * c.RESIZE_FACTOR[0])
width = int(c.IMAGE_SIZE[1] * c.RESIZE_FACTOR[1])

def downsize_image(img):
    return cv2.resize(img, (width, height))


def create_npy(array, filename):
    np.save(c.NPY_PATH + filename, array)
    print('\n' + filename + ' created in: ' + c.NPY_PATH)


def ls(data_path):
    return [os.listdir(data_path), len(os.listdir(data_path))]


def read_image(data_path, image_name):
    return np.array(downsize_image([cv2.imread(os.path.join(data_path, image_name), cv2.IMREAD_GRAYSCALE)][0]))


def preprocess_training_data():
    images, no_of_images = ls(c.TRAIN_DATA_PATH)
    no_of_images = no_of_images / 2

    # create empty arrays of type uint8
    imgs = np.ndarray((no_of_images, 1, height, width), dtype = np.uint8)
    imgs_mask = np.ndarray((no_of_images, 1, height, width), dtype = np.uint8)

    i = 0
    for image_name in images:
        # ignore masks
        if 'mask' in image_name:
            continue

        # set mask name for the current image
        image_mask_name = image_name.split('.')[0] + c.MASK_SUFFIX

        img = read_image(c.TRAIN_DATA_PATH, image_name)
        img_mask = read_image(c.TRAIN_DATA_PATH, image_mask_name)

        imgs[i] = img
        imgs_mask[i] = img_mask

        i += 1
        stdout.write("\r%d" % i + '/' + str(no_of_images) + ' training instances processed.')

    create_npy(imgs, c.TRAIN_NPY)
    create_npy(imgs_mask, c.TRAIN_MASKS_NPY)
    stdout.write("\n")


def preprocess_validation_data():
    images, no_of_images = ls(c.VALIDATION_DATA_PATH)
    no_of_images = no_of_images / 2

    # create empty arrays of type uint8
    imgs = np.ndarray((no_of_images, 1, height, width), dtype = np.uint8)
    imgs_mask = np.ndarray((no_of_images, 1, height, width), dtype = np.uint8)

    i = 0
    for image_name in images:
        # ignore masks
        if 'mask' in image_name:
            continue

        # set mask name for the current image
        image_mask_name = image_name.split('.')[0] + c.MASK_SUFFIX

        img = read_image(c.VALIDATION_DATA_PATH, image_name)

        img_mask = read_image(c.VALIDATION_DATA_PATH, image_mask_name)

        imgs[i] = img
        imgs_mask[i] = img_mask

        i += 1
        stdout.write("\r%d" % i + '/' + str(no_of_images) + ' validation instances processed.')

    create_npy(imgs, c.VALIDATION_NPY)
    create_npy(imgs_mask, c.VALIDATION_MASKS_NPY)
    stdout.write("\n")


def preprocess_testing_data():
    images, no_of_images = ls(c.TEST_DATA_PATH)

    imgs = np.ndarray((no_of_images, 1, height, width), dtype = np.uint8)
    imgs_id = np.ndarray((no_of_images), dtype = np.int32)

    i = 0
    for image_name in images:
        img_id = int(image_name.split('.')[0])

        img = read_image(c.TEST_DATA_PATH, image_name)

        imgs[i] = img
        imgs_id[i] = img_id

        i += 1
        stdout.write("\r%d" % i + '/' + str(no_of_images) + ' test instances processed.')

    create_npy(imgs, c.TEST_NPY)
    create_npy(imgs_id, c.TEST_IDS_NPY)
    stdout.write("\n")


def preprocess_data():
    preprocess_training_data()
    preprocess_validation_data()
    preprocess_testing_data()

if __name__ == '__main__':
    preprocess_data()
