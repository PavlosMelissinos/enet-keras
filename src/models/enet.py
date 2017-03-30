from keras.engine.topology import Input, merge
from keras.models import Model
from keras.layers.core import Activation, Reshape
from keras.utils.visualize_util import plot
import encoder, decoder


def transfer_weights(model, weights=None):
    '''
    Always trains from scratch; never transfers weights
    '''
    print('ENet has found no compatible pretrained weights! Skipping weight transfer...')
    return model


def autoencoder(nc, input_shape,
                loss='categorical_crossentropy',
                optimizer='adadelta'):
    data_shape = input_shape[0] * input_shape[1]
    inp = Input(shape=(input_shape[0], input_shape[1], 3))
    enet = encoder.build(inp)
    enet = decoder.build(enet, nc=nc, in_shape=input_shape)

    enet = Reshape((data_shape, nc), input_shape=(input_shape[0], input_shape[1], nc))(enet)
    enet = Activation('softmax')(enet)
    model = Model(input=inp, output=enet)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'mean_squared_error'])
    name = 'enet'

    return model, name

if __name__ == "__main__":
    autoencoder, name = autoencoder(nc=2, input_shape=(512, 512))
    plot(autoencoder, to_file='{}.png'.format(name), show_shapes=True)
