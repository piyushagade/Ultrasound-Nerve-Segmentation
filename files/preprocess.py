
from __future__ import print_function

import os
import numpy as np

import cv2


# # Global variables

data_path = 'data/raw/'
npy_path = 'data/npy/'
train_data_path = os.path.join(data_path, 'train')
test_data_path = os.path.join(data_path, 'test')
# image dimensions
image_rows = 420
image_cols = 580



# # Create Train data

def create_train_data():
    # calculate total number of images (excluding their corresponding masks) instances.
    no_of_images = len(os.listdir(train_data_path)) / 2

    imgs = np.ndarray((no_of_images, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((no_of_images, 1, image_rows, image_cols), dtype=np.uint8)

    i = 0
    for image_name in os.listdir(train_data_path):
        # skip this iteration if the image is a mask
        if 'mask' in image_name:
            continue

        # set mask name for the current image
        image_mask_name = image_name.split('.')[0] + '_mask.tif'

        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask


        if True:
            print('Processed: {0}/{1} images'.format(i, no_of_images))

        i += 1

    print('Loading done.')

    np.save(npy_path + 'imgs_train.npy', imgs)
    np.save(npy_path + 'imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')



# # Load data

def load_train_data():
    imgs_train = np.load(npy_path + 'imgs_train.npy')
    imgs_mask_train = np.load(npy_path + 'imgs_mask_train.npy')
    return imgs_train, imgs_mask_train


# # Create test data

# In[12]:



def create_test_data():
    images = os.listdir(test_data_path)
    no_of_images = len(images)

    imgs = np.ndarray((no_of_images, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((no_of_images, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)

    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if True:
            print('Processed: {0}/{1} images'.format(i, no_of_images))
        i += 1
    print('Loading done.')

    np.save(npy_path + 'imgs_test.npy', imgs)
    np.save(npy_path + 'imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')



# # Load test data

# In[13]:


def load_test_data():
    imgs_test = np.load(npy_path + 'imgs_test.npy')
    imgs_id = np.load(npy_path + 'imgs_id_test.npy')
    return imgs_test, imgs_id


# # Main function

# In[17]:

if __name__ == '__main__':
    create_train_data()
    create_test_data()

