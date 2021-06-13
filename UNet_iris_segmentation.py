# -*- coding: utf-8 -*-
"""
Created on Fri May 28 15:51:49 2021

@author: Naveen
"""

from keras import optimizers
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Conv2D, Input, UpSampling2D, MaxPooling2D,Concatenate
from keras import backend as K
import util

def load_data():
    mydata = util.dataProcess(240, 320)
    imgs_train, imgs_mask_train = mydata.load_train_data()
    return imgs_train, imgs_mask_train
###########################################################################
def conv_block(x, num_filters):
    x = Conv2D(num_filters, (3, 3),kernel_initializer = 'he_normal', padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3, 3),kernel_initializer = 'he_normal', padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x    



def create_model():
    num_filters = [64,128,256,512]
    inputs = Input(shape=[240, 320, 1])

    skip_x = []
    x = inputs
    ## Encoder
    for f in num_filters:
        x = conv_block(x, f)
        skip_x.append(x)
        x = MaxPooling2D((2, 2))(x)

    ## Bridge
    x = conv_block(x, 1024)

    num_filters.reverse()
    skip_x.reverse()
    ## Decoder
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2))(x)
        xs = skip_x[i]
        x = Concatenate()([x, xs])
        x = conv_block(x, f)

    ## Output
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)

    model = Model(inputs,x)

    model.summary()

    return model
###########################################################################

def train():
    model_path = "C:/Users/Naveen/Desktop/segmentation/project iris/Iris-master/Iris/"

    print("got model")
    model = create_model()
    print("loading data")
    imgs_train, imgs_mask_train = load_data()
    print("loading data done")

    opt = optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

    model_checkpoint = ModelCheckpoint(model_path + 'unet_segmentation.hdf5', monitor='loss', verbose=1,
                                       save_best_only=True, save_weights_only=True, mode='auto', period=1)
    print('Fitting model...')
    early_stop = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='auto')
    lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=6, verbose=0, mode='min', cooldown=0,
                           min_lr=0.000001)
    model.fit(imgs_train, imgs_mask_train, batch_size=2, epochs=5, verbose=1, validation_split=0.2,
                        shuffle=True,
                        callbacks=[model_checkpoint,lr, early_stop])

if __name__ == '__main__':
    train()
    K.clear_session()