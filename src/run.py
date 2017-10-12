# coding=utf-8
from __future__ import absolute_import, division, print_function

from keras import backend as K
import json
import numpy as np
import os
import sys

from src import experiments


def run():
    mode = 'data'
    solver_file = os.path.join('config', 'solver.json')
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    if len(sys.argv) > 2:
        solver_file = sys.argv[2]
        # solver_file = os.path.join('config', solver_file)
        solver_file = os.path.abspath(solver_file)

    print('solver json: {}'.format(solver_file))
    kwargs = json.load(open(solver_file))

    if mode == 'train':
        experiment = experiments.core.SemanticSegmentationExperiment(**kwargs)
    elif mode == 'overfit':
        experiment = experiments.core.OverfittingExperiment(**kwargs)
    elif mode == 'data':
        experiment = experiments.core.DryDatasetExperiment(**kwargs)
    else:
        errmsg = 'This script only supports training at the moment'
        raise NotImplementedError(errmsg)

    experiment.run()


if __name__ == '__main__':
    np.random.seed(1337)  # for reproducibility
    if K.backend() == 'tensorflow':
        print('Tensorflow backend detected; Applying memory usage constraints')
        config = K.tf.ConfigProto(gpu_options=K.tf.GPUOptions(allow_growth=True))
        ss = K.tf.Session(config=config)
        K.set_session(ss)
        ss.run(K.tf.global_variables_initializer())

    run()
