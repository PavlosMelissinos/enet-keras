# coding=utf-8
from keras import backend as K
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Activation, Reshape
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization as BN
from keras.layers.pooling import MaxPooling2D
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.optimizers import Adam, Nadam


def initial_block(x):
    y = Conv2D(filters=64, padding='same', kernel_size=(7, 7), strides=(2, 2))(x)
    y = BN()(y)
    y = Activation('relu')(y)
    y = MaxPooling2D(pool_size=(2, 2))(y)
    return y


def encoder_block(x, filters):
    y = Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        padding='same',
        strides=(2, 2))(x)
    y = BN()(y)
    y = Activation('relu')(y)
    y = Conv2D(filters=filters, padding='same', kernel_size=(3, 3))(y)
    y = BN()(y)

    if K.image_data_format() == 'channels_last':
        actual_filters = K.int_shape(y)[-1]
        residual_filters = K.int_shape(x)[-1]
        actual_spatial_dims = K.int_shape(y)[1:3]
        residual_spatial_dims = K.int_shape(x)[1:3]
    else:
        actual_filters = K.int_shape(y)[1]
        residual_filters = K.int_shape(x)[1]
        actual_spatial_dims = K.int_shape(y)[2:4]
        residual_spatial_dims = K.int_shape(x)[2:4]
    if residual_filters != actual_filters:
        strides = 2 if actual_spatial_dims != residual_spatial_dims else 1
        x = Conv2D(filters=actual_filters, kernel_size=(1, 1), strides=strides)(x)
    elif actual_spatial_dims != residual_spatial_dims:
        x = MaxPooling2D(pool_size=(2, 2))(x)
    added = Add()([x, y])
    added = Activation('relu')(added)

    y = Conv2D(filters=filters, padding='same', kernel_size=(3, 3))(added)
    y = BN()(y)
    y = Activation('relu')(y)
    y = Conv2D(filters=filters, padding='same', kernel_size=(3, 3))(y)
    y = BN()(y)
    y = Activation('relu')(y)
    y = Add()([added, y])

    return y


def decoder_block(x, filters):
    channel_idx = -1 if K.image_data_format() == 'channels_last' else 1
    input_channels = K.int_shape(x)[channel_idx]
    internal_filters = input_channels // 4
    y = Conv2D(filters=internal_filters, padding='same', kernel_size=(1, 1))(x)
    y = BN()(y)
    y = Activation('relu')(y)
    y = Conv2DTranspose(
        filters=internal_filters,
        kernel_size=(3, 3),
        padding='same',
        strides=(2, 2))(y)
    y = BN()(y)
    y = Activation('relu')(y)
    y = Conv2D(filters=filters, kernel_size=(1, 1), padding='same')(y)
    y = BN()(y)
    y = Activation('relu')(y)
    return y


def final_block(x, num_classes):
    y = Conv2DTranspose(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        strides=2)(x)
    y = BN()(y)
    y = Activation('relu')(y)
    y = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(y)
    y = BN()(y)
    y = Activation('relu')(y)
    y = Conv2DTranspose(
        filters=num_classes,
        kernel_size=(2, 2),
        padding='same',
        strides=2)(y)
    return y


def build(nc, w, h,
          loss='categorical_crossentropy',
          # optimizer='adadelta'):
          optimizer='adam',
          metrics=None,
          **kwargs):
    name = 'linknet'
    inp = Input(shape=(h, w, 3), name='image')
    encoder = initial_block(inp)
    encoder1 = encoder_block(x=encoder, filters=64)
    encoder2 = encoder_block(x=encoder1, filters=128)
    encoder3 = encoder_block(x=encoder2, filters=256)
    encoder4 = encoder_block(x=encoder3, filters=512)

    decoder4 = decoder_block(x=encoder4, filters=256)
    decoder = Add()([encoder3, decoder4])
    decoder3 = decoder_block(x=decoder, filters=128)
    decoder = Add()([encoder2, decoder3])
    decoder2 = decoder_block(x=decoder, filters=64)
    decoder = Add()([encoder1, decoder2])
    decoder1 = decoder_block(x=decoder, filters=64)
    decoder = final_block(x=decoder1, num_classes=nc)

    hw = K.int_shape(decoder)[1] * K.int_shape(decoder)[2]
    target_shape = (hw, nc)
    decoder = Reshape(target_shape=target_shape)(decoder)
    decoder = Activation('softmax', name='output')(decoder)

    model = Model(inputs=inp, outputs=decoder)

    if metrics is None:
        metrics = ['accuracy']
    # optimizer = Adam(lr=0.01)
    # optimizer = Nadam()
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model, name


def transfer_weights(model, **kwargs):
    return model


class LinkNet(object):
    def __init__(self, h, w, nc):
        self.h = h
        self.w = w
        self.nc = nc

    def build(self):
        return build(self.h, self.w, self.nc)
