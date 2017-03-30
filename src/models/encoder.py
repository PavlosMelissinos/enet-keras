from keras.engine.topology import merge
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import AtrousConvolution2D, Convolution2D, ZeroPadding2D
from keras.layers.core import Permute, SpatialDropout2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D


def initial_block(inp, nb_filter=13, nb_row=3, nb_col=3, conv_stride=(2, 2)):
    conv = Convolution2D(nb_filter, nb_row, nb_col, border_mode='same', subsample=conv_stride)(inp)
    max_pool = MaxPooling2D()(inp)
    merged = merge([conv, max_pool], mode='concat', concat_axis=3)
    return merged


def bottleneck(inp, output, internal_scale=4, use_relu=True, asymmetric=0, dilated=0, downsample=False, dropout_rate=0.1):
    # main branch
    internal = output / internal_scale
    encoder = inp

    ## 1x1
    input_stride = 2 if downsample else 1  # the first 1x1 projection is replaced with a 2x2 convolution when downsampling
    encoder = Convolution2D(internal, input_stride, input_stride, border_mode='same', subsample=(input_stride, input_stride), bias=False)(encoder)
    ## Batch normalization + PReLU
    encoder = BatchNormalization(momentum=0.1)(encoder) # enet uses momentum of 0.1, keras default is 0.99
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    ## conv
    if not asymmetric and not dilated:
        encoder = Convolution2D(nb_filter=internal, nb_row=3, nb_col=3, border_mode='same')(encoder)
    elif asymmetric:
        encoder = Convolution2D(nb_filter=internal, nb_row=1, nb_col=asymmetric, border_mode='same', bias=False)(encoder)
        encoder = Convolution2D(nb_filter=internal, nb_row=asymmetric, nb_col=1, border_mode='same')(encoder)
    elif dilated:
        encoder = AtrousConvolution2D(nb_filter=internal, nb_row=3, nb_col=3, atrous_rate=(dilated, dilated), border_mode='same')(encoder)
    else:
        raise(Exception('You shouldn\'t be here'))

    ## Batch normalization + PReLU
    encoder = BatchNormalization(momentum=0.1)(encoder) # enet uses momentum of 0.1, keras default is 0.99
    encoder = PReLU(shared_axes=[1, 2])(encoder)
    
    ## 1x1
    encoder = Convolution2D(nb_filter=output, nb_row=1, nb_col=1, border_mode='same', bias=False)(encoder)
    
    ## Batch normalization + Spatial dropout
    encoder = BatchNormalization(momentum=0.1)(encoder) # enet uses momentum of 0.1, keras default is 0.99
    encoder = SpatialDropout2D(dropout_rate)(encoder)

    other = inp
    # other branch
    if downsample:
        other = MaxPooling2D()(other)
        other = Permute((1, 3, 2))(other)
        pad_featmaps = output - inp.get_shape().as_list()[3]
        other = ZeroPadding2D(padding=(0, 0, 0, pad_featmaps))(other)
        other = Permute((1, 3, 2))(other)

    encoder = merge([encoder, other], mode='sum')
    encoder = PReLU(shared_axes=[1, 2])(encoder)
    return encoder


def build(inp, dropout_rate=0.01):
    enet = initial_block(inp)
    enet = bottleneck(enet, 64, downsample=True, dropout_rate=dropout_rate)  # bottleneck 1.0
    for i in range(4):
        enet = bottleneck(enet, 64, dropout_rate=dropout_rate) # bottleneck 1.i
    
    enet = bottleneck(enet, 128, downsample=True)  # bottleneck 2.0
    # bottleneck 2.x and 3.x
    for i in range(2):
        enet = bottleneck(enet, 128)  # bottleneck 2.1
        enet = bottleneck(enet, 128, dilated=2)  # bottleneck 2.2
        enet = bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.3
        enet = bottleneck(enet, 128, dilated=4)  # bottleneck 2.4
        enet = bottleneck(enet, 128)  # bottleneck 2.5
        enet = bottleneck(enet, 128, dilated=8)  # bottleneck 2.6
        enet = bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.7
        enet = bottleneck(enet, 128, dilated=16)  # bottleneck 2.8
    return enet

