# coding=utf-8
from keras.layers import Input
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.layers.merge import Concatenate
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.core import Activation, Reshape


def build(nc, w, h,
          loss='categorical_crossentropy',
          # optimizer='adadelta'):
          optimizer=None,
          metrics=None,
          **kwargs):
    inputs = Input(shape=(w, h, 3), name='image')

    '''
    unet with crop(because padding = valid) 
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'valid', 
    kernel_initializer = 'he_normal')(inputs)
    print "conv1 shape:",conv1.shape
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'valid', 
    kernel_initializer = 'he_normal')(conv1)
    print "conv1 shape:",conv1.shape
    crop1 = Cropping2D(cropping=((90,90),(90,90)))(conv1)
    print "crop1 shape:",crop1.shape
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print "pool1 shape:",pool1.shape
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'valid', 
    kernel_initializer = 'he_normal')(pool1)
    print "conv2 shape:",conv2.shape
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'valid', 
    kernel_initializer = 'he_normal')(conv2)
    print "conv2 shape:",conv2.shape
    crop2 = Cropping2D(cropping=((41,41),(41,41)))(conv2)
    print "crop2 shape:",crop2.shape
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print "pool2 shape:",pool2.shape
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'valid', 
    kernel_initializer = 'he_normal')(pool2)
    print "conv3 shape:",conv3.shape
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'valid', 
    kernel_initializer = 'he_normal')(conv3)
    print "conv3 shape:",conv3.shape
    crop3 = Cropping2D(cropping=((16,17),(16,17)))(conv3)
    print "crop3 shape:",crop3.shape
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    print "pool3 shape:",pool3.shape
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'valid', 
    kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'valid', 
    kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    crop4 = Cropping2D(cropping=((4,4),(4,4)))(drop4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'valid', 
    kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'valid', 
    kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', 
    kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = merge([crop4,up6], mode = 'concat', concat_axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'valid', 
    kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'valid', 
    kernel_initializer = 'he_normal')(conv6)
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', 
    kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = merge([crop3,up7], mode = 'concat', concat_axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'valid', 
    kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'valid', 
    kernel_initializer = 'he_normal')(conv7)
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', 
    kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = merge([crop2,up8], mode = 'concat', concat_axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'valid', 
    kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'valid', 
    kernel_initializer = 'he_normal')(conv8)
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', 
    kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = merge([crop1,up9], mode = 'concat', concat_axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'valid', 
    kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'valid', 
    kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'valid', 
    kernel_initializer = 'he_normal')(conv9)
    '''

    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs)
    print("conv1 shape:", conv1.shape)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    print("conv1 shape:", conv1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print("pool1 shape:", pool1.shape)

    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    print("conv2 shape:", conv2.shape)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    print("conv2 shape:", conv2.shape)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print("pool2 shape:", pool2.shape)

    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    print("conv3 shape:", conv3.shape)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    print("conv3 shape:", conv3.shape)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    print("pool3 shape:", pool3.shape)

    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = Concatenate()([drop4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate()([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate()([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate()([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(nc, 1)(conv9)

    hw = K.int_shape(conv10)[1] * K.int_shape(conv10)[2]
    target_shape = (hw, nc)
    decoder = Reshape(target_shape=target_shape)(conv10)

    decoder = Activation('softmax', name='output')(decoder)


    model = Model(input=inputs, output=decoder)

    if optimizer is None:
        optimizer = Adam(lr=1e-4)
    if metrics is None:
        metrics = ['accuracy']

    model.compile(optimizer=optimizer, loss=loss,
                  metrics=metrics)

    name = 'unet'

    return model, name


def transfer_weights(model, **kwargs):
    return model


class UNet(object):
    def __init__(self, h, w, nc):
        self.h = h
        self.w = w
        self.nc = nc

    def build(self):
        return build(self.h, self.w, self.nc)
