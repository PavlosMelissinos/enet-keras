# coding=utf-8
from __future__ import absolute_import, division, print_function

from keras import backend as K
import json
import numpy as np
import os

from src.experiments.core import Experiment


def old_dataset_loader(solver):
    from src.data import datasets
    data_config = solver['data_managers']

    # transfer target dimensions to data configuration dictionary
    data_config['default']['h'] = solver['dh']
    data_config['default']['w'] = solver['dw']
    dataset_name = data_config['dataset_name']

    dataset_class = datasets.load(dataset_name=dataset_name)

    data_config['data_type'] = data_config['train_data_type']
    train_dataset = dataset_class(config=data_config)

    data_config['data_type'] = data_config['val_data_type']
    val_dataset = dataset_class(config=data_config)
    return train_dataset, val_dataset


if __name__ == '__main__':
    np.random.seed(1337)  # for reproducibility
    if K.backend() == 'tensorflow':
        print('Tensorflow backend detected; Applying memory usage constraints')
        ss = K.tf.Session(config=K.tf.ConfigProto(gpu_options=K.tf.GPUOptions(allow_growth=True)))
        K.set_session(ss)
        ss.run(K.tf.global_variables_initializer())

    solver_json = 'config/solver.json'
    print('solver json: {}'.format(os.path.abspath(solver_json)))

    experiment = Experiment(solver=json.load(open(solver_json)))
    experiment.run()
