# coding=utf-8
from __future__ import absolute_import, print_function

from keras.engine.topology import Input
from keras.layers.core import Activation, Reshape
from keras.models import Model
from keras.utils import plot_model

from . import encoder
from . import decoder


def transfer_weights(model, weights=None):
    """
    Always trains from scratch; never transfers weights
    :param model: 
    :param weights:
    :return: 
    """
    print('ENet has found no compatible pretrained weights! Skipping weight transfer...')
    return model


def build(nc, w, h,
          loss='categorical_crossentropy',
          optimizer='adadelta'):
    # data_shape = input_shape[0] * input_shape[1] if input_shape and None not in input_shape else None
    data_shape = w * h if None not in (w, h) else -1  # TODO: -1 or None?
    inp = Input(shape=(h, w, 3))
    out = encoder.build(inp)
    out = decoder.build(inp=inp, encoder=out, nc=nc)

    out = Reshape((data_shape, nc))(out)  # TODO: need to remove data_shape for multi-scale training
    out = Activation('softmax')(out)
    model = Model(inputs=inp, outputs=out)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'mean_squared_error'])
    name = 'icnet'

    return model, name

if __name__ == "__main__":
    autoencoder, name = build(nc=19, w=1025, h=2049)
    plot_model(autoencoder, to_file='{}.png'.format(name), show_shapes=True)
