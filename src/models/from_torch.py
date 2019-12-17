# coding=utf-8
from __future__ import absolute_import, division, print_function

import numpy as np
import os
import pickle as pkl
import torchfile
from src.definitions import PROJECT_ROOT

CONV_TRANSPOSE = (2, 3, 1, 0)


def from_torch(torch_model):
    def expand_module(module):
        if 'weight' in module._obj or b'weight' in module._obj:
            return [module]
        if 'modules' in module._obj or b'modules' in module._obj:
            # return module._obj[b'modules']
            lst = [expand_module(submodule) for submodule in module._obj[b'modules']]
            return [sublist for item in lst for sublist in item]
        return [None]

    enet = torchfile.load(filename=torch_model)
    all_enet_modules = [module for module in expand_module(enet) if module is not None]
    all_enet_modules = [module for module in all_enet_modules if b'weight' in module._obj]

    weights = []
    # for module in all_enet_modules:
    for module in all_enet_modules:
        item = {}
        if module.torch_typename() == b'cudnn.SpatialConvolution':
            item['weight'] = module[b'weight']
            if b'bias' in module._obj:
                item['bias'] = module[b'bias']
        elif module.torch_typename() == b'nn.SpatialBatchNormalization':
            item = {
                'gamma': module[b'weight'],
                'beta': module[b'bias'],
                'moving_mean': module[b'running_mean'],
                'moving_variance': module[b'running_var'],
            }
        elif module.torch_typename() == b'nn.PReLU':
            weight = np.expand_dims(np.expand_dims(module[b'weight'], 0), 0)
            item['weight'] = weight
        elif module.torch_typename() == b'nn.SpatialDilatedConvolution':
            item['weight'] = module[b'weight']
            if b'bias' in module._obj:
                item['bias'] = module[b'bias']
        elif module.torch_typename() == b'nn.SpatialFullConvolution':
            item['weight'] = module[b'weight']
            if b'bias' in module._obj:
                item['bias'] = module[b'bias']
        else:
            print('Unhandled torch layer: {}'.format(module.torch_typename()))
        item['torch_typename'] = module.torch_typename().decode()

        if 'Convolution' in item['torch_typename']:
            item['weight'] = np.transpose(item['weight'], CONV_TRANSPOSE)

        weights.append(item)
    return weights


if __name__ == "__main__":
    torch_model = PROJECT_ROOT / 'pretrained' / 'model-best.net'
    weights = from_torch(torch_model=torch_model)
    # weights = [module['weight'] for module in all_enet_modules]
    with (PROJECT_ROOT / 'pretrained' / 'torch_enet.pkl').open(mode='wb') as fout:
        pkl.dump(obj=weights, file=fout)
