from __future__ import print_function

import cv2
import keras
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.preprocessing.image import ImageDataGenerator

from modules import preprocess as pp
from modules import data_split as ds
from modules import constants as c

import matplotlib.pyplot as plt

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

img_rows = int(c.IMAGE_SIZE[0] * c.RESIZE_FACTOR[0])
img_cols = int(c.IMAGE_SIZE[1] * c.RESIZE_FACTOR[1])

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((1, 64, 80))

    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation = 'relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation = 'relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation = 'relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation = 'relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation = 'relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation = 'relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation = 'relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation = 'relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation = 'relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation = 'relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation = 'relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation = 'relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation = 'relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation = 'relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation = 'relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation = 'relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation = 'relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def preprocess():
    pp.preprocess_data()

def augment(img_train):
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    datagen.fit(img_train)

    return datagen

def train_and_predict():

    print('Loading and preprocessing train data...')

    [imgs_train, imgs_mask_train] = ds.load_train_data()

    imgs_train = imgs_train.astype('float32')

    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('Creating and compiling model...')
    model = get_unet()

    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', save_best_only=True)


    print('Fitting model...')
    # validate_model(1)

    datagen = augment(imgs_train)

    model.fit_generator(datagen.flow(imgs_train, imgs_mask_train, batch_size=32),
                        samples_per_epoch=len(imgs_train), nb_epoch = 1)

    # Fit data into model
    # model.fit(imgs_train, imgs_mask_train, batch_size=32, nb_epoch=20, verbose=1, shuffle=True,
    #           callbacks=[model_checkpoint])

    # Display history
    # summarize history for accuracy
    plt.plot(model.history['acc'])
    plt.plot(model.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig('historyvsaccura.png')

    # summarize history for loss
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig('historyvsloss.png')

    # Predict
    print('Loading and preprocessing test data...')

    imgs_test, imgs_id_test = ds.load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std


    print('Loading saved weights...')
    model.load_weights('unet.hdf5')

    print('Predicting masks on test data...')
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)


# def validate_model(nfolds):
#     from sklearn.cross_validation import cross_val_score
#
#     model = create_model(img_rows, img_cols)
#     X_train, X_valid = train_data[train_index], train_data[test_index]
#     Y_train, Y_valid = train_target[train_index], train_target[test_index]
#
#     num_fold += 1
#     print('Start KFold number {} from {}'.format(num_fold, nfolds))
#     print('Split train: ', len(X_train), len(Y_train))
#     print('Split valid: ', len(X_valid), len(Y_valid))
#
#     model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#           shuffle=True, verbose=2, validation_data=(X_valid, Y_valid),
#           callbacks=callbacks)
#
#     predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
#     score = log_loss(Y_valid, predictions_valid)
#     print('Score log_loss: ', score)
#
#     # Store valid predictions
#     for i in range(len(test_index)):
#         yfull_train[test_index[i]] = predictions_valid[i]
#
#     # Store test predictions
#     test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)


if __name__ == '__main__':
    # preprocess()
    train_and_predict()
