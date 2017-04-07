from keras.layers.convolutional import Convolution2D, Deconvolution2D, UpSampling2D
from keras.layers.core import Activation
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization


def bottleneck(encoder, output, upsample=False, reverse_module=False):
    internal = output / 4
    input_stride = 2 if upsample else 1
    
    x = Convolution2D(internal, (input_stride, input_stride), padding='same', use_bias=False)(encoder)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)
    if not upsample:
        x = Convolution2D(internal, (3, 3), padding='same', use_bias=True)(x)
    else:
        b, w, h, nb_filters = encoder.get_shape().as_list()
        in_shape = x.get_shape().as_list()
        # x = Deconvolution2D(internal, 3, 3, output_shape=(None, w * 2, h * 2, internal), border_mode='same',
        #                     subsample=(2, 2), input_shape=in_shape)(x)
        x = Deconvolution2D(internal, (3, 3), padding='same', strides=(2, 2), input_shape=in_shape)(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)

    x = Convolution2D(output, (1, 1), padding='same', use_bias=False)(x)

    other = encoder
    if encoder.get_shape()[-1] != output or upsample:
        other = Convolution2D(output, (1, 1), padding='same', use_bias=False)(other)
        other = BatchNormalization(momentum=0.1)(other)
        if upsample and reverse_module:
            other = UpSampling2D(size=(2, 2))(other)
        
    if not upsample or reverse_module:
        x = BatchNormalization(momentum=0.1)(x)
    else:
        return x
    
    decoder = add([x, other])
    decoder = Activation('relu')(decoder)
    return decoder


def build(encoder, nc, in_shape, dropout_rate=0.1):
    # print(encoder.get_shape().as_list())
    enet = bottleneck(encoder, 64, upsample=True, reverse_module=True)  # bottleneck 4.0
    # print(enet.get_shape().as_list())
    # import sys
    # sys.exit()
    enet = bottleneck(enet, 64)  # bottleneck 4.1
    enet = bottleneck(enet, 64)  # bottleneck 4.2
    enet = bottleneck(enet, 16, upsample=True, reverse_module=True)  # bottleneck 5.0
    enet = bottleneck(enet, 16)  # bottleneck 5.1

    # out_shape = enet.get_shape().as_list()
    # out_shape = [out_shape[0], 2 * out_shape[1], 2 * out_shape[2], nc]
    # enet = Deconvolution2D(nc, 2, 2, output_shape=out_shape, border_mode='same', subsample=(2, 2),
    #                        input_shape=in_shape)(enet)
    enet = Deconvolution2D(nc, (2, 2), padding='same', strides=(2, 2), input_shape=in_shape)(enet)
    return enet

