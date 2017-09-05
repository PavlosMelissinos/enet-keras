# coding=utf-8
from __future__ import absolute_import, print_function

from keras.engine.topology import Input
from keras.layers.core import Activation, Reshape
from keras.models import Model
from keras.utils import plot_model
from src.models.enet_unpooling import encoder, decoder
import os
import pickle as pkl


def transfer_weights(model, weights=None, keep_top=False):
    """
    Transfers weights from torch-enet if they are available as {PROJECT_ROOT}/models/pretrained/torch_enet.pkl after
    running from_torch.py.

    :param keep_top: Skips the final Transpose Convolution layer if False.
    :param model: the model to copy the weights to.
    :param weights: the filename that contains the set of layers to copy. Run from_torch.py first.
    :return: a model that contains the updated weights. This function mutates the contents of the input model as well.
    """

    def special_cases(idx):
        """
        :param idx: original index of layer
        :return: the corrected index of the layer as well as the corresponding layer
        """

        if idx == 266:
            actual_idx = 267
        elif idx == 267:
            actual_idx = 268
        elif idx == 268:
            actual_idx = 266

        elif idx == 299:
            actual_idx = 300
        elif idx == 300:
            actual_idx = 301
        elif idx == 301:
            actual_idx = 299
        else:
            actual_idx = idx
        return actual_idx, model.layers[actual_idx]

    if weights is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.join(dir_path, os.pardir, os.pardir, os.pardir)
        weights = os.path.join(project_root, 'models', 'pretrained', 'torch_enet.pkl')
    if not os.path.isfile(weights):
        print('ENet has found no compatible pretrained weights! Skipping weight transfer...')
    with open(weights, 'rb') as fin:
        weights_mem = pkl.load(fin)
        idx = 0
        for num, layer in enumerate(model.layers):
            actual_num, layer = special_cases(num)  # special cases due to non-matching layer sequences

            if not layer.weights:
                continue

            item = weights_mem[idx]
            layer_name = item['torch_typename']
            if layer_name == 'cudnn.SpatialConvolution':
                if 'bias' in item:
                    new_values = [item['weight'], item['bias']]
                else:
                    new_values = [item['weight']]
            elif layer_name == 'nn.SpatialBatchNormalization':
                new_values = [item['gamma'], item['beta'], item['moving_mean'], item['moving_variance']]
            elif layer_name == 'nn.PReLU':
                new_values = [item['weight']]
            elif layer_name == 'nn.SpatialDilatedConvolution':
                if 'bias' in item:
                    new_values = [item['weight'], item['bias']]
                else:
                    new_values = [item['weight']]
            elif layer_name == 'nn.SpatialFullConvolution':
                new_values = layer.get_weights()
                if keep_top:
                    if 'bias' in item:
                        new_values = [item['weight'], item['bias']]
                    else:
                        new_values = [item['weight']]
            else:
                print(layer_name)
                new_values = layer.get_weights()
            layer.set_weights(new_values)
            idx += 1
    return model


def build(nc, w, h,
          loss='categorical_crossentropy',
          optimizer='adadelta'):
    data_shape = w * h if None not in (w, h) else -1  # TODO: -1 or None?
    inp = Input(shape=(h, w, 3))
    enet = encoder.build(inp)
    enet = decoder.build(enet, nc=nc)
    name = 'enet_unpooling'

    enet = Reshape((data_shape, nc))(enet)  # TODO: need to remove data_shape for multi-scale training

    enet = Activation('softmax')(enet)
    model = Model(inputs=inp, outputs=enet)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'mean_squared_error'])

    return model, name


def main():
    nc = 81
    dw = 256
    dh = 256
    dir_path = os.path.dirname(os.path.realpath(__file__))
    target_path = os.path.join(dir_path, 'model.png')

    autoencoder, model_name = build(nc=nc, w=dw, h=dh)
    plot_model(autoencoder, to_file=target_path, show_shapes=True)
    transfer_weights(model=autoencoder)

if __name__ == "__main__":
    main()